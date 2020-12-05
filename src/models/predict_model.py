import os
import sys
path_file = os.path.abspath(__file__)
path_utils = os.path.join(os.path.dirname(os.path.dirname(path_file)), 'utils')
sys.path.insert(0, path_utils)

from io_dataset import read_from_txt
from tokenizer import load_tokenizer, transform_tokenizer
from utils import get_maxlen, padding_doc
import re
import argparse

import numpy as np

from tensorflow.keras.models import load_model

def parse_arguments():
    parser = argparse.ArgumentParser(description='Making the dataset.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to input')
    parser.add_argument('--output_path', type=str, help='Path to output')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to input')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model trained')
    parser.add_argument('--n_sentences', type=int, default=2, help='Path to model trained')
    return parser.parse_args()

def clean_text(docs):
    print('Clean text...')
    # Add space around puntuation
    docs = [re.sub(r"([?.!,])", r" \1 ", sentences) for sentences in docs]
    
    # Remove all character expect ("a-z", "A-Z", "0-9", ",.!?")
    docs = [re.sub(r"[^a-zA-Z?!.,0-9]+", " ", sentences).strip() for sentences in docs]

    return docs


def processing_data(samples):
    origin = samples.copy()
    full = clean_text(samples)
    
    print('Processing...')
    # Split sentence in each sample of full text
    full = [[sentence.strip() 
                for sentence in  re.split('\?|\.|!', sentences)
                    if 1<len(sentence.strip())] 
                for sentences in full]
                
    origin = [[sentence.strip() 
                for sentence in  re.split('\?|\.|!', sentences)
                    if 1<len(sentence.strip())] 
                for sentences in origin]

    return full, origin


def main():
    # Get arguments
    args = parse_arguments()
    
    # Get input
    txt = read_from_txt(args.input_path)
    
    # Processing
    processed, origin = processing_data(txt)
    origin = origin[0]
    
    # Tokenizer
    token = load_tokenizer(args.tokenizer_path)
    x = transform_tokenizer(token, processed)
    
    # Load model
    model = load_model(args.model_path)
    
    maxlen_sentence, maxlen_word = model.input.shape[1:]
    x = padding_doc(x, maxlen_sentence, maxlen_word)
    
    # Predict
    pred = model.predict(x)[0][:len(origin)]    
    pred_sort = pred.argsort()[-args.n_sentences:]
    
    # print({idx:i for idx, i in enumerate(pred)})
    # print(len(origin), sorted(pred_sort))
    
    result = '. '.join([origin[i] for i in sorted(pred_sort)])
    
    print('\n\nFull text:', txt[0])
    print('\n\nSummary:', pred_sort, result)
    
    

if __name__ == "__main__":
    main()
