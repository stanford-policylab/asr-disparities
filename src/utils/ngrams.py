import os
import sys
import pandas as pd
import numpy as np
from jiwer import wer
from nltk import ngrams
import math

def split_white_black(all_trans):
# Separate into AAVE and White samples
    black_stack = all_trans[all_trans['race_ethnicity']=='Black']
    white_stack = all_trans[all_trans['race_ethnicity']=='White']
    return black_stack, white_stack

def create_ngrams(df, n):
# Create n-grams of phrases from given transcriptions
    all_grams = []
    for row in df.to_records():
        grams = ngrams(row.clean_content.split(), n)
        for gram in grams:
            all_grams.append({
                'segment_filename': row.segment_filename,
                'basefile': row.basefile,
                'race_ethnicity': row.race_ethnicity,
                'age': row.age,
                'gender': row.gender,
                'n': n,
                'ngram': ' '.join(gram)
            })
    return pd.DataFrame(all_grams)

def best_phrase(ground_truth, hyp):
# Finds the phrase in the hypothesis string that best matches the ground truth.
# In the case of ties, returns the shortest match
    ground_truth_len = len(ground_truth.split())
    try:
        hyp_tokens = hyp.split()
    except:
        print(hyp)
        raise
    
    opt_phrase = ""
    opt_phrase_len = math.inf
    opt_wer = math.inf
    
    for start_pos in range(len(hyp_tokens)):
        for end_pos in range(start_pos, start_pos+ground_truth_len+1):    
            this_phrase = " ".join(hyp_tokens[start_pos:end_pos])
            this_phrase_len = len(this_phrase.split())
            this_wer = wer(ground_truth, this_phrase)
                    
            if (this_wer < opt_wer) or (this_wer == opt_wer and this_phrase_len < opt_phrase_len):
                opt_phrase = this_phrase
                opt_phrase_len = this_phrase_len
                opt_wer = this_wer
        
    return(opt_phrase, opt_wer)
    
if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Command example: python3 ngrams.py <input_folder>")
        exit(1)    
            
    base_folder = sys.argv[1]

    # Import previously generated transcript WER values (unmatched)
    clean_transcripts_wer = pd.read_csv(base_folder + 'input/CORAAL_transcripts.csv')
    clean_transcripts_wer = clean_transcripts_wer.replace(np.nan, '', regex=True)

    clean_black_stack, clean_white_stack = split_white_black(clean_transcripts_wer)

    # Generates n-grams of size n until no more common n-grams are found
    all_grams = pd.DataFrame()
    make_grams = True
    n = 5
    while make_grams:
        white_grams = create_ngrams(clean_white_stack, n)
        black_grams = create_ngrams(clean_black_stack, n)
        commons = set(white_grams.ngram) & set(black_grams.ngram)
        white_grams = white_grams[white_grams.ngram.isin(commons)]
        black_grams = black_grams[black_grams.ngram.isin(commons)]
        all_grams = pd.concat([all_grams, white_grams, black_grams])
        
        make_grams = len(commons) > 0
        n += 1

    # Removes n-grams that are substrings of larger n-grams
    is_substring = {}
    for n in sorted(all_grams.n.unique()):
        longer_grams = all_grams[all_grams.n > n]
        for ngram, segment_filename in all_grams[all_grams.n == n][['ngram', 'segment_filename']].values:
            superstrings = longer_grams[(longer_grams.ngram.str.contains(ngram))\
                                        & (longer_grams.segment_filename == segment_filename)]
            is_substring[(segment_filename, ngram)] = len(superstrings) > 0
            
    not_substring = [not is_substring[(sf, g)] for g, sf in all_grams[['ngram', 'segment_filename']].values]
    all_grams = all_grams[not_substring]
    all_grams.gender = all_grams.gender.str.lower()

    # Matches ngrams by gender and age, then samples 100 of each race
    matched_grams = []
    black_grams = all_grams[all_grams.race_ethnicity == 'Black']
    white_grams = all_grams[all_grams.race_ethnicity == 'White']
    age_diff = 5
    sample_size = 100

    for row in black_grams.to_records():
        match = white_grams[(white_grams.ngram == row.ngram)\
                            & (white_grams.gender == row.gender)\
                            & ((white_grams.age - row.age).abs() <= age_diff)]
        if len(match) > 0:
            matched_grams.append(row)
            matched_grams += list(match.to_records())
    matched_grams = pd.DataFrame(np.array(matched_grams))
    matched_grams = matched_grams.drop('index', axis=1).drop_duplicates(subset=['basefile', 'ngram'])

    matched_grams = matched_grams.drop_duplicates(subset=['race_ethnicity', 'ngram'])

    # Merge ngrams to full WER dataset

    ngrams_with_asr = clean_transcripts_wer.merge(matched_grams,
                                                  on = ['segment_filename',
                                                        'basefile',
                                                        'age', 
                                                        'race_ethnicity'], 
                                                         how = 'inner')

    # Find best phrase for each ASR

    clean_asr_trans_list = ['clean_google',
                            'clean_ibm',
                            'clean_amazon',
                            'clean_msft',
                            'clean_apple']

    for clean_asr in clean_asr_trans_list:
        temp_ngram = ngrams_with_asr.apply(lambda x: best_phrase(x['ngram'], x[clean_asr]), axis=1)
        ngrams_with_asr.loc[:, clean_asr+'_phrase'] = temp_ngram.map(lambda x: x[0])
        ngrams_with_asr.loc[:, clean_asr+'_ngram_wer'] = temp_ngram.map(lambda x: x[1])

    # Export n-gram WERs
    ngrams_with_asr.to_csv(base_folder + 'output/ngrams_with_asr_wer.csv')

    # Get average ASRs by race
    ngram_race_breakdown = pd.DataFrame()
    ngrams_without_inf = ngrams_with_asr.replace([np.inf], 1.0)
    for clean_asr in clean_asr_trans_list:
        x = ngrams_without_inf.groupby(['race_ethnicity'])[clean_asr+'_ngram_wer'].mean().to_frame().transpose()
        ngram_race_breakdown = pd.concat([ngram_race_breakdown, x])
        
    # Get standard errors
    ngram_race_se = pd.DataFrame()
    for clean_asr in clean_asr_trans_list:
        x = ngrams_without_inf.groupby(['race_ethnicity'])[clean_asr+'_ngram_wer'].sem().to_frame().transpose()
        ngram_race_se = pd.concat([ngram_race_se, x])
        
    # Make cleaned table of average n-gram WERs by race
    ngram_race_breakdown['ASR'] = ['Google', 'IBM', 'Amazon', 'Microsoft', 'Apple']
    ngram_race_breakdown = ngram_race_breakdown.reset_index(level=0, drop=True)
    ngram_race_breakdown = ngram_race_breakdown.reindex([4,1,0,2,3])
    ngram_race_breakdown['Average AAVE WER'] = round(ngram_race_breakdown['Black'],2)
    ngram_race_breakdown['Average White WER'] = round(ngram_race_breakdown['White'],2)

    ngram_race_breakdown = ngram_race_breakdown.drop(['Black', 'White'], axis=1)
    ngram_race_breakdown = ngram_race_breakdown.reset_index(level=0, drop=True)

    # Generate LaTeX table of average n-gram WERs as Table 2
    print(ngram_race_breakdown.to_latex(index=False))
