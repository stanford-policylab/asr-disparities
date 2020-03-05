import math
import argparse
import logging
import pandas as pd
import re
import torch
from tqdm import tqdm, trange
from utils import *
from pytorch_transformers import *

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

def score_gpt(model, tokenizer, content):
    # set model to run on GPUs if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # compute total loss for content
    total_len, total_loss = 0, 0
    for phrase in tqdm(content):
        tokenize_input = tokenizer.tokenize(phrase)
        token_ids = tokenizer.convert_tokens_to_ids(tokenize_input)
        
        seq_len = len(tokenize_input) - 1
        total_len += seq_len
    
        with torch.no_grad():
            data = torch.tensor([token_ids]).to(device)
            outputs = model(data, labels=data)
            loss = outputs[0]
            total_loss += loss * seq_len
            
    # convert to perplexity
    perplexity = math.exp(total_loss/total_len)
    return perplexity
    
def score_transfo(model, tokenizer, content):
    # set model to run on GPUs if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # formatter for transformer-xl model
    # expects punctuation to be separated by spaces
    space_pat = re.compile(r"([,.?'\"])")
    def space_str(s):
        s = space_pat.sub(" \\1", s)
        return(s)
 
    # compute total loss for content
    total_len, total_loss = 0, 0
    for phrase in tqdm(content):
        phrase = space_str(phrase)
        tokenize_input = tokenizer.tokenize(phrase)
        token_ids = tokenizer.convert_tokens_to_ids(tokenize_input)
        
        seq_len = len(tokenize_input) - 1
        total_len += seq_len
    
        with torch.no_grad():
            data = torch.tensor([token_ids[:-1]]).to(device)
            target = torch.tensor([token_ids[1:]]).to(device)
            loss, _, mems = model(data, target)
            total_loss += loss.sum()
            
    # convert to perplexity
    perplexity = math.exp(total_loss/total_len)
    return perplexity

def entropy(content):
    tokens = {}
    for phrase in content:
        tks = phrase.split()
        for t in tks:
            if t in tokens:
                tokens[t] += 1
            else:
                tokens[t] = 1
            
    num_tokens = sum(tokens.values())
    num_distinct = len(tokens)
    token_dist = [t/num_tokens for t in tokens.values()]
    ent = -sum([p*math.log(p) for p in token_dist])
    perplexity = math.exp(ent)
    
    return(num_tokens, num_distinct, ent, perplexity)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Language modeling')
    parser.add_argument('--analysis', type=str, default='entropy',
                        choices=['entropy', 'perplexity', 'copula'],
                        help='type of analysis to perform')
    args = parser.parse_args()
    
    # load data
    transcripts_file = relpath('../output/CORAAL_WER.csv')
    # to run below, transcripts_file should instead contain matched snippets with both CORAAL and VOC transcripts
    transcripts = pd.read_csv(transcripts_file, index_col=0)
    aave = transcripts.loc[transcripts['black_flag'] == 1, 'apostrophe_clean_content']
    voc = transcripts.loc[transcripts['black_flag'] == 0, 'apostrophe_clean_content']
    aave_clean = transcripts.loc[transcripts['black_flag'] == 1, 'clean_content']
    voc_clean = transcripts.loc[transcripts['black_flag'] == 0, 'clean_content']

    # compute overall entropy for AAVE and white speech
    if args.analysis == 'entropy':
        # output header
        print("\t".join(["variety", "num_tokens", "num_distinct", "entropy", "perplexity"]))
        
        # overall entropy for white content
        num_tokens, num_distinct, ent, perplexity = entropy(voc_clean)
        print("\t".join(map(str, ["white", num_tokens, num_distinct, ent, perplexity])))

        # overall entropy for AAVE content        
        num_tokens, num_distinct, ent, perplexity = entropy(aave_clean)
        print("\t".join(map(str, ["AAVE", num_tokens, num_distinct, ent, perplexity])))

    # compute perplexity for various language models
    elif args.analysis == 'perplexity':
        # output header
        print("\t".join(["model", "variety", "perplexity"]))

        # perplexity under GPT-2
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        print("\t".join(map(str, ["GPT-2", "white", score_gpt(model, tokenizer, voc)])))
        print("\t".join(map(str, ["GPT-2", "AAVE", score_gpt(model, tokenizer, aave)])))
        
        # perplexity under GPT
        model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
        tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        print("\t".join(map(str, ["GPT", "white", score_gpt(model, tokenizer, voc)])))
        print("\t".join(map(str, ["GPT", "AAVE", score_gpt(model, tokenizer, aave)])))
        
        # perplexity under Transformer-XL
        model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
        tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
        print("\t".join(map(str, ["Transformer-XL", "white", score_transfo(model, tokenizer, voc)])))
        print("\t".join(map(str, ["Transformer-XL", "AAVE", score_transfo(model, tokenizer, aave)])))
        
    # compute perplexity for example phrases with and without copula drops
    elif args.analysis == 'copula':
        # output header
        print("\t".join(["perplexity", "phrase"]))

        # example phrases
        content = ['We going to the arc.',
                   'We are going to the arc.',
                   "We're going to the arc.",
                   'He a pastor.',
                   'He is a pastor.',
                   "He's a pastor.",
                   'We able to fight for the cause.',
                   'We are able to fight for the cause.',
                   "We're able to fight for the cause.",
                   'Where they from?',
                   'Where are they from?',
                   'Have you decided what you going to sing?',
                   'Have you decided what you gonna sing?',
                   'Have you decided what you are going to sing?',
                   "Have you decided what you're going to sing?",
                   "Have you decided what you're gonna sing?",
                   "Y'all got to fix the rules to get them where they supposed to belong.",
                   "Y'all got to fix the rules to get them where they are supposed to belong.",
                   "Y'all got to fix the rules to get them where they're supposed to belong."]
    
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        for phrase in content:
            print("\t".join(map(str, [score_gpt(model, tokenizer, [phrase]), phrase])))
