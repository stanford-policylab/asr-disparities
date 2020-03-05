import os
import sys
import pandas as pd
import numpy as np
import collections

def split_white_black(all_trans):
    # Separate into AAVE and White samples
    black_stack = all_trans[all_trans['black_flag']==1]
    white_stack = all_trans[all_trans['black_flag']==0]
    return black_stack, white_stack

def find_word_counts(transcript, column):
    # Get total count of cleaned words (including repeated words) in CORAAL and VOC
    transcript['words'] = transcript.apply(lambda x: x[column].split(), axis=1)
    word_sum = transcript.words.sum()
    output=collections.Counter(word_sum)
    df = pd.DataFrame.from_dict(output, orient='index').reset_index()
    df.columns = ['word', column+'_count']
    print(column, len(df))
    return df
    
if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Command example: python3 ngrams.py <input_folder>")
        exit(1)    
            
    base_folder = sys.argv[1]

    # Import previously generated transcript WER values (matched)
    clean_transcripts_wer_only = pd.read_csv(base_folder + 'output/matched_wer.csv')
    clean_transcripts = pd.read_csv(base_folder + 'output/CORAAL_transcripts.csv')
    clean_transcripts_wer = clean_transcripts_wer_only.merge(clean_transcripts,
                                                             on = 'segment_filename')
    
    clean_transcripts_wer = clean_transcripts_wer.replace(np.nan, '', regex=True)

    clean_black_stack, clean_white_stack = split_white_black(clean_transcripts_wer)

    count_coraal_words = find_word_counts(clean_black_stack, 'clean_content')
    count_voc_words = find_word_counts(clean_white_stack, 'clean_content')

    # Find word counts in each ASR lexicon (where 'word' column contains unique words)
    unique_google_words = find_word_counts(clean_transcripts_wer, 'clean_google')
    unique_amazon_words = find_word_counts(clean_transcripts_wer, 'clean_amazon')
    unique_msft_words = find_word_counts(clean_transcripts_wer, 'clean_msft')
    unique_ibm_words = find_word_counts(clean_transcripts_wer, 'clean_ibm')
    unique_apple_words = find_word_counts(clean_transcripts_wer, 'clean_apple')

    # Merge to match CORAAL / VOC words to ASR lexicon
    coraal_merge_words = count_coraal_words
    voc_merge_words = count_voc_words

    for df in [unique_google_words, unique_amazon_words, unique_msft_words, unique_ibm_words, unique_apple_words]:
        coraal_merge_words = coraal_merge_words.merge(df, on = 'word', how = 'left')
        voc_merge_words = voc_merge_words.merge(df, on = 'word', how = 'left')

    # Get word sum
    total_coraal_words = coraal_merge_words['clean_content_count'].sum()
    total_voc_words = voc_merge_words['clean_content_count'].sum()

    asr_list = ['apple', 'ibm', 'google', 'amazon', 'msft']
    coraal_list = []
    voc_list = []

    for asr in asr_list:
        voc_words_in_corpus = voc_merge_words['clean_content_count'].where(
            ~voc_merge_words['clean_'+asr+'_count'].isna()).sum()
        voc_list.append(voc_words_in_corpus/total_voc_words)
        
        coraal_words_in_corpus = coraal_merge_words['clean_content_count'].where(
            ~coraal_merge_words['clean_'+asr+'_count'].isna()).sum()
        coraal_list.append(coraal_words_in_corpus/total_coraal_words)

    asr_list_cap = ['Apple', 'IBM', 'Google', 'Amazon', 'Microsoft']
    coraal_list_round = [str(round(100*elem, 2))+'%' for elem in coraal_list]    
    voc_list_round = [str(round(100*elem, 2))+'%' for elem in voc_list]
    pd.DataFrame(list(zip(asr_list_cap, coraal_list_round, voc_list_round)), columns = ['ASR', 
                                                                        'CORAAL % Words in ASR Corpus', 
                                                                        'VOC % Words in ASR Corpus'])

    # Sanity check the count of Google ASR words

    print(total_coraal_words) # Total number of words uttered by black speakers in our sample
    print(coraal_merge_words['clean_content_count'].where(~coraal_merge_words['clean_google_count'].isna()).sum()) # Words that Google had in ASR from CORAAL

    print(total_voc_words) # Total number of words uttered by black speakers in our sample
    print(voc_merge_words['clean_content_count'].where(~voc_merge_words['clean_google_count'].isna()).sum()) # Words that Google had in ASR from VOC

    # Find set intersection and differences between VOC and CORAAL

    black_words = count_coraal_words['word']
    white_words = count_voc_words['word']

    black_not_in_white = black_words[~((black_words.isin(white_words)))]
    white_not_in_black = white_words[~((white_words.isin(black_words)))]
    white_black_intersection = white_words[((white_words.isin(black_words)))]

    print(len(black_not_in_white), len(white_not_in_black), len(white_black_intersection))
    print(len(black_words), len(white_words))

    # Find set intersection and differences between VOC and CORAAL

    black_not_in_white_counts = pd.DataFrame(black_not_in_white).merge(count_coraal_words, on = 'word')
    black_not_in_white_counts.columns = ['word', 'coraal_count']
    black_not_in_white_counts = black_not_in_white_counts.sort_values('coraal_count', ascending=False)

    white_not_in_black_counts = pd.DataFrame(white_not_in_black).merge(count_voc_words, on = 'word')
    white_not_in_black_counts.columns = ['word', 'voc_count']
    white_not_in_black_counts = white_not_in_black_counts.sort_values('voc_count', ascending=False)

    white_black_intersection_counts = (pd.DataFrame(white_black_intersection).merge(count_coraal_words, on = 'word')).merge(count_voc_words, on='word')
    white_black_intersection_counts.columns = ['word', 'coraal_count', 'voc_count']
    white_black_intersection_counts['total_count'] = white_black_intersection_counts['coraal_count'] + white_black_intersection_counts['voc_count']
    white_black_intersection_counts = white_black_intersection_counts.sort_values('total_count', ascending=False)
    white_black_intersection_counts

    print(len(black_not_in_white_counts), len(white_not_in_black_counts), len(white_black_intersection_counts))
