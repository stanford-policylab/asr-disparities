import os
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

p = inflect.engine()
t2d = text2digits.Text2Digits()

# Full cleaning functions
def remove_markers(line, markers):
    # Remove any text within markers, e.g. 'We(BR) went' -> 'We went'
    # markers = list of pairs, e.g. ['()', '[]'] denoting breath or noise in transcripts
    for s, e in markers:
         line = re.sub(" ?\\" + s + "[^" + e + "]+\\" + e, "", line)
    return line

def clean_coraal(baseline_snippets):
    # Clean CORAAL human transcript
    # Restrict to CORAAL rows
    baseline_coraal = baseline_snippets[baseline_snippets['race_ethnicity']=='Black']
    
    # Replace original unmatched CORAAL transcript square brackets with squiggly bracket
    baseline_coraal.loc[:,'clean_content'] = baseline_coraal.loc[:,'content'].copy()
    baseline_coraal.loc[:,'clean_content'] = baseline_coraal['clean_content'].str.replace('\[','\{')
    baseline_coraal.loc[:,'clean_content'] = baseline_coraal['clean_content'].str.replace('\]','\}')
    
    def clean_within_coraal(text):

        # Relabel CORAAL words. For consideration: aks -> ask?
        split_words = text.split()
        split_words = [x if x != 'busses' else 'buses' for x in split_words]
        split_words = [x if x != 'aks' else 'ask' for x in split_words]
        split_words = [x if x != 'aksing' else 'asking' for x in split_words]
        split_words = [x if x != 'aksed' else 'asked' for x in split_words]
        text = ' '.join(split_words)
        
        # remove CORAAL unintelligible flags
        text = re.sub("\/(?i)unintelligible\/",'',''.join(text))
        text = re.sub("\/(?i)inaudible\/",'',''.join(text))
        text = re.sub('\/RD(.*?)\/', '',''.join(text))
        text = re.sub('\/(\?)\1*\/', '',''.join(text))
        
        # remove nonlinguistic markers
        text = remove_markers(text, ['<>', '()', '{}'])

        return text

    baseline_coraal['clean_content'] = baseline_coraal.apply(lambda x: clean_within_coraal(x['clean_content']), axis=1)
    
    return baseline_coraal

def clean_voc(baseline_snippets):
    # Clean VOC human transcript
    # Restrict to VOC rows
    baseline_voc = baseline_snippets[baseline_snippets['race_ethnicity']=='White']
    
    pre_list = ['thier', 'humbolt', 'arcada', 'ninteen', 'marajuana', 'theatre', 'portugeuse', 'majorca']
    post_list = ['their', 'Humboldt', 'Arcata', 'nineteen', 'marijuana', 'theater', 'portuguese', 'mallorca']
    def clean_within_voc(text):

        # Relabel misspellings
        split_words = text.split()
        for i in range(len(pre_list)):
            split_words = [x if x.lower() != pre_list[i] else post_list[i] for x in split_words]
        text = ' '.join(split_words)   
        #new_words = [x if x != 'thier' else 'their' for x in split_words]

        # remove nonlinguistic markers
        text = remove_markers(text, ['<>', '{}'])

        return text

    baseline_voc['clean_content'] = baseline_voc.apply(lambda x: clean_within_voc(x['content']), axis=1)
    
    return baseline_voc

# Standardize state abbreviations
states = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY',
}

def fix_state_abbrevs(text):
    # Standardize state abbreviations
    ix = 0
    state_result = []
    wordlist = text.split()
    while ix < len(wordlist):
        word = wordlist[ix].lower().capitalize()
        if word in states.keys(): # is this correct check?
            new_word = states[word]
        elif (ix < len(wordlist)-1) and ((word + ' ' + wordlist[ix+1].lower().capitalize()) in states.keys()):
            new_word = states[(word + ' ' + wordlist[ix+1].lower().capitalize())]
            ix += 1
        else:
            new_word = word
        state_result.append(new_word)
        ix += 1
    text = ' '.join(state_result)
    return text

def fix_numbers(text):
    # Standardize number parsing and dollars
    split_words_num = text.split()
    new_list = []
    for i in range(len(split_words_num)):
        x = split_words_num[i]
        
        # deal with years
        if x.isdigit():
            if (1100 <= int(x) < 2000) or (2010 <= int(x) < 2100) or (int(x) == 5050):
                # deal with years as colloquially spoken
                new_word = p.number_to_words(x[:2]) + " " + p.number_to_words(x[2:])
            elif "and" in p.number_to_words(x):
                # remove 'and' from e.g. 'four hundred and ninety five'
                output = p.number_to_words(x)
                resultwords  = [word for word in output.split() if word not in ['and']]    
                new_word = ' '.join(resultwords)
            else:
                new_word = p.number_to_words(x)
            
        # deal with cases like 1st, 2nd, etc.
        elif re.match(r"(\d+)(\w+)", x, re.I):
            single_digits = ['1st', '2nd', '3rd', '5th', '8th', '9th']
            double_digits = ['12th']
            single_num = ['1', '2', '3', '5', '8', '9']
            double_num = ['12']
            single_digit_labels = ['first', 'second', 'third', 'fifth', 'eighth', 'ninth']
            double_digit_labels = ['twelfth']
            all_digits = single_digits + double_digits
            all_labels = single_digit_labels + double_digit_labels
            if x in all_digits:
                new_word = all_labels[all_digits.index(x)]
            else:
                items = re.match(r"(\d+)(\w+)", x, re.I).groups()
                if (items[1] not in ['s', 'th', 'st', 'nd', 'rd']):
                    new_word = fix_numbers(items[0]) + " " + items[1]
                elif (items[0][-2:] in double_num):
                    new_word = fix_numbers(str(100*int(items[0][:-2]))) + " " + fix_numbers(items[0][-2:]+items[1])
                elif ((items[0][-1:] in single_num) and items[0][-2:-1] != '1'):
                    try:
                        new_word = fix_numbers(str(10*int(items[0][:-1]))) + " " + fix_numbers(items[0][-1:]+items[1])
                    except:
                        new_word = fix_numbers(items[0]) + items[1]
                # deal with case e.g. 80s
                elif (items[1] in ['s', 'th']) and (p.number_to_words(items[0])[-1] == 'y'):
                    new_word = fix_numbers(items[0])[:-1] + "ie" + items[1]
                else:
                    new_word = fix_numbers(items[0]) + items[1]
                    
        # deal with dollars
        elif re.match(r"\$[^\]]+", x, re.I):
            # deal with $ to 'dollars'
            money = fix_numbers(x[1:])
            if x[1:] in ["1", "a"]:
                new_word = money + " dollar"
            else:
                new_word = money + " dollars"
                
        elif re.match(r"\£[^\]]+", x, re.I):
            # deal with £ to 'pounds'
            money = fix_numbers(x[1:])
            if x[1:] in ["1", "a"]:
                new_word = money + " pound"
            else:
                new_word = money + " pounds"
                
        else:
            new_word = x       
        
        new_list.append(new_word)
        
    text = ' '.join(new_list)
    text =re.sub(r'[^\s\w$]|_', ' ',text)
    
    # Deal with written out years (two thousand and ten -> twenty ten)
    for double_dig in range(10, 100):
        double_dig_str = p.number_to_words(double_dig)
        text = re.sub('two thousand and ' + double_dig_str, 'twenty ' + double_dig_str, text.lower())
        text = re.sub('two thousand ' + double_dig_str, 'twenty ' + double_dig_str, text.lower())

    # Change e.g. 101 to 'one oh one' -- good for area codes
    single_dig_list = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    for j in single_dig_list:
        text = re.sub('thousand and ' + j, 'thousand ' + j, text.lower())
        for k in single_dig_list:
            #print(j + ' hundred ' + k)
            text = re.sub(j + ' hundred ' + k + ' ', j + ' oh ' + k + ' ', text.lower())
            text = re.sub(j + ' hundred ' + k + '$', j + ' oh ' + k, text.lower())
    
    text = re.sub("\s+"," ",''.join(text)) # standardize whitespace
    
    return text

def clean_all_transcripts(baseline_snippets):
    # Function for text cleaning on all ASR transcriptions, as well as human transcriptions of VOC and CORAAL
    new_baseline = baseline_snippets.copy()
    new_baseline['google_transcription'] = new_baseline['google_transcription'].replace(np.nan, '', regex=True)  
    new_baseline['ibm_transcription'] = new_baseline['ibm_transcription'].replace(np.nan, '', regex=True)  
    new_baseline['amazon_transcription'] = new_baseline['amazon_transcription'].replace(np.nan, '', regex=True)  
    new_baseline['msft_transcription'] = new_baseline['msft_transcription'].replace(np.nan, '', regex=True)  
    new_baseline['apple_transcription'] = new_baseline['apple_transcription'].replace(np.nan, '', regex=True)  
    
    swear_words = ['nigga', 'niggas', 'shit', 'bitch', 'damn', 'fuck', 'fuckin', 'fucking', 'motherfuckin', 'motherfucking']
    filler_words = ['um', 'uh', 'mm', 'hm', 'ooh', 'woo', 'mhm', 'huh', 'ha']
    
    pre_cardinal = ['N', 'E', 'S', 'W', 'NE', 'NW', 'SE', 'SW']
    post_cardinal = ['North', 'East', 'South', 'West', 'Northeast', 'Northwest', 'Southeast', 'Southwest']
    
    pre_list = ['cuz', 'ok', 'o', 'till', 'yup', 'imma', 'mister', 'doctor',
                'gonna', 'tryna',
               'carryout', 'sawmill', 'highschool', 'worldclass',
               'saint', 'street', 'state',
                'avenue', 'road', 'boulevard',
               'theatre', 'neighbour', 'neighbours', 'neighbourhood', 'programme']
    post_list = ['cause', 'okay', 'oh', 'til', 'yep', 'ima', 'mr', 'dr',
                 'going to', 'trying to',
                'carry out', 'saw mill', 'high school', 'world class',
                 'st', 'st', 'st',
                 'ave', 'rd', 'blvd',
                 'theater', 'neighbor', 'neighbors', 'neighborhood', 'program']

    def clean_within_all(text):
        
        # remove hesitation from IBM transcript
        text = re.sub('%HESITATION',' ',''.join(text))
        
        # fix spacing in certain spellings
        text = re.sub('T V','TV',''.join(text))
        text = re.sub('D C','DC',''.join(text))
        
        # remove remaining floating non-linguistic words
        single_paren = ['<','>', '(',')', '{','}','[',']']
        for paren in single_paren:
            linguistic_words  = [word for word in text.split() if paren not in word]    
            text = ' '.join(linguistic_words)
              
        # general string cleaning
        text = re.sub(r"([a-z])\-([a-z])", r"\1 \2", text , 0, re.IGNORECASE) # replace inter-word hyphen with space
        text = re.sub("'",'',''.join(text)) # remove apostrophe
        text =re.sub(r'[^\s\w$]|_', ' ',text) # replace special characters with space, except $
        text = re.sub("\s+"," ",''.join(text)) # standardize whitespace
        
        # update numeric numbers to strings and remove $
        text = re.sub("ft ²", "square feet", ''.join(text))
        text = fix_numbers(text)
        text = re.sub("\$",'dollars',''.join(text))
        text = re.sub("\£",'pounds',''.join(text))
        
        # standardize spellings
        split_words = text.split()
        for i in range(len(pre_list)):
            split_words = [x if x.lower() != pre_list[i] else post_list[i] for x in split_words]
        text = ' '.join(split_words)        
        
        # deal with cardinal directions
        split_words_dir = text.split()
        for i in range(len(pre_cardinal)):
            split_words_dir = [x if x != pre_cardinal[i] else post_cardinal[i] for x in split_words_dir]
        text = ' '.join(split_words_dir)
        
        # deal with state abbreviations
        text = fix_state_abbrevs(text)
        text = text.lower()
   
        # update spacing in certain spellings
        spacing_list_pre = ['north east', 'north west', 'south east', 'south west', 'all right']
        spacing_list_post = ['northeast', 'northwest', 'southeast', 'southwest', 'alright']
        for i in range(len(spacing_list_pre)):
            text = re.sub(spacing_list_pre[i], spacing_list_post[i],''.join(text))

        # remove filler words and swear words
        remove_words = swear_words + filler_words
        resultwords  = [word for word in text.split() if word not in remove_words]    
        result = ' '.join(resultwords)
        
        return result
    
    new_baseline['clean_content'] = new_baseline.apply(lambda x: clean_within_all(x['clean_content']), axis=1)
    new_baseline['clean_google'] = new_baseline.apply(lambda x: clean_within_all(x['google_transcription']), axis=1)
    new_baseline['clean_ibm'] = new_baseline.apply(lambda x: clean_within_all(x['ibm_transcription']), axis=1)
    new_baseline['clean_amazon'] = new_baseline.apply(lambda x: clean_within_all(x['amazon_transcription']), axis=1)
    new_baseline['clean_msft'] = new_baseline.apply(lambda x: clean_within_all(x['msft_transcription']), axis=1)
    new_baseline['clean_apple'] = new_baseline.apply(lambda x: clean_within_all(x['apple_transcription']), axis=1)

    return new_baseline

# Partial cleaning functions for perplexity calculation: maintain cases and punctuation (apostrophes matter a lot)

def apostrophe_clean_coraal(baseline_snippets):
    # Partially clean CORAAL human transcript -- ignore cases & punctuation
    baseline_coraal = baseline_snippets
    # Replace original unmatched CORAAL transcript square brackets with squiggly bracket
    baseline_coraal.loc[:,'apostrophe_clean_content'] = baseline_coraal.loc[:,'content'].copy()
    baseline_coraal.loc[:,'apostrophe_clean_content'] = baseline_coraal['apostrophe_clean_content'].str.replace('\[','\{')
    baseline_coraal.loc[:,'apostrophe_clean_content'] = baseline_coraal['apostrophe_clean_content'].str.replace('\]','\}')
    
    def apostrophe_clean_within_coraal(text):

        # Relabel CORAAL words. For consideration: aks -> ask?
        split_words = text.split()
        split_words = [x if x != 'busses' else 'buses' for x in split_words]
        split_words = [x if x != 'aks' else 'ask' for x in split_words]
        split_words = [x if x != 'aksing' else 'asking' for x in split_words]
        split_words = [x if x != 'aksed' else 'asked' for x in split_words]
        text = ' '.join(split_words)
        
        # remove CORAAL unintelligible flags
        text = re.sub("\/(?i)unintelligible\/",'',''.join(text))
        text = re.sub("\/(?i)inaudible\/",'',''.join(text))
        text = re.sub('\/RD(.*?)\/', '',''.join(text))
        text = re.sub('\/(\?)\1*\/', '',''.join(text))
        
        # remove nonlinguistic markers
        text = remove_markers(text, ['<>', '()', '{}'])

        return text

    baseline_coraal['apostrophe_clean_content'] = baseline_coraal.apply(lambda x: apostrophe_clean_within_coraal(x['apostrophe_clean_content']), axis=1)
    
    return baseline_coraal

def apostrophe_clean_voc(baseline_snippets):
    # Partially clean VOC human transcript
    # Restrict to CORAAL rows
    baseline_voc = baseline_snippets
    
    pre_list = ['thier', 'humbolt', 'arcada', 'ninteen', 'marajuana', 'theatre', 'portugeuse', 'majorca']
    post_list = ['their', 'Humboldt', 'Arcata', 'nineteen', 'marijuana', 'theater', 'portuguese', 'mallorca']
    def apostrophe_clean_within_voc(text):

        # Relabel misspellings
        split_words = text.split()
        for i in range(len(pre_list)):
            split_words = [x if x.lower() != pre_list[i] else post_list[i] for x in split_words]
        text = ' '.join(split_words)   
        #new_words = [x if x != 'thier' else 'their' for x in split_words]

        # remove nonlinguistic markers
        text = remove_markers(text, ['<>', '{}'])

        return text

    baseline_voc['apostrophe_clean_content'] = baseline_voc.apply(lambda x: apostrophe_clean_within_voc(x['content']), axis=1)
    
    return baseline_voc

def apostrophe_fix_numbers(text):
    # Partially clean numbers
    split_words_num = text.split()
    new_list = []
    for i in range(len(split_words_num)):
        x = split_words_num[i]
        
        # deal with years
        if x.isdigit():
            if (1100 <= int(x) < 2000) or (2010 <= int(x) < 2100) or (int(x) == 5050):
                # deal with years as colloquially spoken
                new_word = p.number_to_words(x[:2]) + " " + p.number_to_words(x[2:])
            elif "and" in p.number_to_words(x):
                # remove 'and' from e.g. 'four hundred and ninety five'
                output = p.number_to_words(x)
                resultwords  = [word for word in output.split() if word not in ['and']]    
                new_word = ' '.join(resultwords)
            else:
                new_word = p.number_to_words(x)
            
        # deal with cases like 1st, 2nd, etc.
        elif re.match(r"(\d+)(\w+)", x, re.I):
            single_digits = ['1st', '2nd', '3rd', '5th', '8th', '9th']
            double_digits = ['12th']
            single_num = ['1', '2', '3', '5', '8', '9']
            double_num = ['12']
            single_digit_labels = ['first', 'second', 'third', 'fifth', 'eighth', 'ninth']
            double_digit_labels = ['twelfth']
            all_digits = single_digits + double_digits
            all_labels = single_digit_labels + double_digit_labels
            if x in all_digits:
                new_word = all_labels[all_digits.index(x)]
            else:
                items = re.match(r"(\d+)(\w+)", x, re.I).groups()
                if (items[1] not in ['s', 'th', 'st', 'nd', 'rd']):
                    new_word = fix_numbers(items[0]) + " " + items[1]
                elif (items[0][-2:] in double_num):
                    new_word = fix_numbers(str(100*int(items[0][:-2]))) + " " + fix_numbers(items[0][-2:]+items[1])
                elif ((items[0][-1:] in single_num) and items[0][-2:-1] != '1'):
                    try:
                        new_word = fix_numbers(str(10*int(items[0][:-1]))) + " " + fix_numbers(items[0][-1:]+items[1])
                    except:
                        new_word = fix_numbers(items[0]) + items[1]
                # deal with case e.g. 80s
                elif (items[1] in ['s', 'th']) and (p.number_to_words(items[0])[-1] == 'y'):
                    new_word = fix_numbers(items[0])[:-1] + "ie" + items[1]
                else:
                    new_word = fix_numbers(items[0]) + items[1]
                    
        # deal with dollars
        elif re.match(r"\$[^\]]+", x, re.I):
            # deal with $ to 'dollars'
            money = fix_numbers(x[1:])
            if x[1:] in ["1", "a"]:
                new_word = money + " dollar"
            else:
                new_word = money + " dollars"
        else:
            new_word = x       
        
        new_list.append(new_word)
        
    text = ' '.join(new_list)
    text =re.sub(r'[^\s\w$\'\.\?\,\!]|_', ' ',text)
    
    # Deal with written out years (two thousand and ten -> twenty ten)
    for double_dig in range(10, 100):
        double_dig_str = p.number_to_words(double_dig)
        text = re.sub('two thousand and ' + double_dig_str, 'twenty ' + double_dig_str, text)
        text = re.sub('two thousand ' + double_dig_str, 'twenty ' + double_dig_str, text)
        text = re.sub('Two thousand and ' + double_dig_str, 'Twenty ' + double_dig_str, text)
        text = re.sub('Two thousand ' + double_dig_str, 'Twenty ' + double_dig_str, text)

    # Change e.g. 101 to 'one oh one' -- good for area codes
    single_dig_list = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    for j in single_dig_list:
        text = re.sub('thousand and ' + j, 'thousand ' + j, text.lower())
        for k in single_dig_list:
            #print(j + ' hundred ' + k)
            text = re.sub(j + ' hundred ' + k + ' ', j + ' oh ' + k + ' ', text.lower())
            text = re.sub(j + ' hundred ' + k + '$', j + ' oh ' + k, text.lower())
    
    text = re.sub("\s+"," ",''.join(text)) # standardize whitespace
    
    return text

def apostrophe_fix_state_abbrevs(text):
    # Partially clean all for state abbreviations with punctuation and original uppercasing
    ix = 0
    state_result = []
    wordlist = text.split()
    while ix < len(wordlist):
        orig_word = wordlist[ix]
        word = wordlist[ix].lower().capitalize()
        if word in states.keys():
            new_word = states[word]
        elif (ix < len(wordlist)-1) and ((word + ' ' + wordlist[ix+1].lower().capitalize()) in states.keys()):
            new_word = states[(word + ' ' + wordlist[ix+1].lower().capitalize())]
            ix += 1
        else:
            new_word = orig_word
        state_result.append(new_word)
        ix += 1
    text = ' '.join(state_result)
    return text

def apostrophe_clean_all_transcripts(baseline_snippets):
    # Partially clean all transcripts
    new_baseline = baseline_snippets.copy()
    
    swear_words = ['nigga', 'niggas', 'shit', 'bitch', 'damn', 'fuck', 'fuckin', 'fucking', 'motherfuckin', 'motherfucking']
    filler_words = ['um', 'uh', 'mm', 'hm', 'ooh', 'woo', 'mhm', 'huh', 'ha']
    
    pre_cardinal = ['N', 'E', 'S', 'W', 'NE', 'NW', 'SE', 'SW']
    post_cardinal = ['North', 'East', 'South', 'West', 'Northeast', 'Northwest', 'Southeast', 'Southwest']
    
    pre_list = ['cuz', 'ok', 'o', 'till', 'yup', 'imma', 'mister', 'doctor',
                'gonna', 'tryna',
               'carryout', 'sawmill', 'highschool', 'worldclass',
               'theatre', 'neighbour', 'neighbours', 'neighbourhood', 'programme']
    post_list = ['cause', 'okay', 'oh', 'til', 'yep', 'ima', 'mr', 'dr',
                 'going to', 'trying to',
                'carry out', 'saw mill', 'high school', 'world class',
                 'theater', 'neighbor', 'neighbors', 'neighborhood', 'program']

    def apostrophe_clean_within_all(text):
        
        # remove hesitation from IBM transcript
        text = re.sub('%HESITATION',' ',''.join(text))
        
        # fix spacing in certain spellings
        text = re.sub('T V','TV',''.join(text))
        text = re.sub('D C','DC',''.join(text))
        
        # remove remaining floating non-linguistic words
        single_paren = ['<','>', '(',')', '{','}','[',']']
        for paren in single_paren:
            linguistic_words  = [word for word in text.split() if paren not in word]    
            text = ' '.join(linguistic_words)
              
        # general string cleaning
        text = re.sub(r"([a-z])\-([a-z])", r"\1 \2", text , 0, re.IGNORECASE) # replace inter-word hyphen with space
        #text = re.sub("'",'',''.join(text)) # remove apostrophe
        text =re.sub(r'[^\s\w$\'\.\?\,\!]|_', ' ',text) # replace special characters with space, except $ and apostrophe
        text = re.sub("\s+"," ",''.join(text)) # standardize whitespace
        
        # update numeric numbers to strings and remove $
        text = re.sub("ft ²", "square feet", ''.join(text))
        #text = apostrophe_fix_numbers(text)
        text = re.sub("\$",'dollars',''.join(text))
        
        # standardize spellings
        split_words = text.split()
        for i in range(len(pre_list)):
            split_words = [x if re.sub('\,','',x.lower()) != pre_list[i] else post_list[i] for x in split_words]
        text = ' '.join(split_words)        
        
        # deal with cardinal directions
        split_words_dir = text.split()
        for i in range(len(pre_cardinal)):
            split_words_dir = [x if re.sub('\,','',x) != pre_cardinal[i] else post_cardinal[i] for x in split_words_dir]
        text = ' '.join(split_words_dir)
        
        # deal with state abbreviations
        text = apostrophe_fix_state_abbrevs(text)
        #text = text.lower()
   
        # update spacing in certain spellings
        spacing_list_pre = ['north east', 'north west', 'south east', 'south west', 'all right']
        spacing_list_post = ['northeast', 'northwest', 'southeast', 'southwest', 'alright']
        for i in range(len(spacing_list_pre)):
            text = re.sub(spacing_list_pre[i], spacing_list_post[i],''.join(text))

        # remove filler words and swear words
        remove_words = swear_words + filler_words
        resultwords  = [word for word in text.split() if re.sub('\,','',word.lower()) not in remove_words]
        #resultwords  = [word for word in text.split() if word.lower()[:-1] not in remove_words]
        result = ' '.join(resultwords)
        
        # capitalize first word, remove extra space before comma
        result = re.sub("\s+\,",",",''.join(result))
        result = result[0].capitalize() + result[1:]
        if result[-1] not in ['.','?','!',',']:
            result = result + '.'
        if result[-1] == ',':
            result = result[:-1] + '.'
        
        result = re.sub("\s+\.",".",''.join(result))
        result = re.sub("\s+\!","!",''.join(result))
        result = re.sub("\s+\?","?",''.join(result))
        
        result = re.sub(" Cause"," cause",''.join(result))
        
        return result
    
    new_baseline['apostrophe_clean_content'] = new_baseline.apply(lambda x: apostrophe_clean_within_all(x['apostrophe_clean_content']), axis=1)

    return new_baseline
