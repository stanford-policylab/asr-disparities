import os
import sys
import glob
import pandas as pd
import numpy as np
import re
import inflect
from text2digits import text2digits
from jiwer import wer
from nltk import ngrams
import math
import collections
from transcript_cleaning_functions import *

p = inflect.engine()
t2d = text2digits.Text2Digits()

def relpath(relp):
    # contruct path relative to location of current script
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, relp)
    return(filename)

def clean_everything(df):
    # Apply cleaning rules from Transcript_Cleaning_Functions.py to CORAAL and VOC
    all_usable = clean_coraal(df)
    clean_all = clean_all_transcripts(all_usable)
    return clean_all

def wer_calc(transcripts, human_clean_col, asr_clean_col):
    # Calculate WER
    new_transcripts = transcripts.copy()
    ground_truth = transcripts[human_clean_col].tolist()
    for col in asr_clean_col:
        new_transcripts[col] = new_transcripts[col].replace(np.nan, '', regex=True)
        asr_trans = new_transcripts[col].tolist()
        wer_list = []
        for i in range(len(ground_truth)):
            wer_list.append(wer(ground_truth[i], asr_trans[i]))
        new_transcripts[col+"_wer"] = wer_list
    return new_transcripts

if __name__ == '__main__':  

    # Apply cleaning rules
    usable_snippets = pd.read_csv(relpath('../input/CORAAL_transcripts.csv'))
    clean_usable_snippets = clean_everything(usable_snippets)

    # Post-cleaning restriction to snippets that have more than 5 words
    clean_usable_snippets['wordcount'] = clean_usable_snippets['clean_content'].str.split().str.len()
    clean_usable_snippets = clean_usable_snippets[clean_usable_snippets['wordcount']>=5]
    print(len(clean_usable_snippets))

    # Create ASR list for WER calculations
    clean_asr_trans_list = ['clean_google',
                            'clean_ibm',
                            'clean_amazon',
                            'clean_msft',
                            'clean_apple']

    # Run WER calculations on all usable snippets, with cleaning
    clean_transcripts_wer = wer_calc(clean_usable_snippets, 'clean_content', clean_asr_trans_list)

    # Apply punctuation-ignoring cleaning rules (for human transcriptions) to the same subset of snippets as determined above
    punctuation_coraal = apostrophe_clean_coraal(clean_transcripts_wer)
    punctuation_clean_snippets = apostrophe_clean_all_transcripts(punctuation_coraal)

    # Export WER (with punctuation-less transcript cleaning for LM perplexity calculation)
    punctuation_clean_snippets.to_csv(relpath('../output/CORAAL_WER.csv'), index=False)
