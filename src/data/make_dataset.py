import os
import sys
path_file = os.path.abspath(__file__)
path_utils = os.path.join(os.path.dirname(os.path.dirname(path_file)), 'utils')
sys.path.insert(0, path_utils)

from io_dataset import save_processed, read_from_txt

import re
import gc
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Making the dataset.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to input')
    parser.add_argument('--output_path', type=str, required=True, help='Path to output')
    parser.add_argument('--maxlen_sentence', type=int, default=25, help='Limit number sentence of each sample')
    parser.add_argument('--maxlen_word', type=int, default=80, help='Limit number word of each sentence')
    return parser.parse_args()


def read_data(folder_data):
    full = read_from_txt(os.path.join(folder_data, 'full.txt'))
    summ = read_from_txt(os.path.join(folder_data, 'summ.txt'))
    return full, summ


def clean_text(docs):
    print('Clean text...')
    # Add space around puntuation
    docs = [re.sub(r"([?.!,])", r" \1 ", sentences) for sentences in docs]
    
    # Remove all character expect ("a-z", "A-Z", "0-9", ",.!?")
    docs = [re.sub(r"[^a-zA-Z?!.,0-9]+", " ", sentences).strip() for sentences in docs]

    return docs


def processing_data(samples):
    print('Processing...')
    full = clean_text(samples[0])
    summ = clean_text(samples[1])
    
    # Split sentence in each sample of full text
    full = [[sentence.strip() 
                for sentence in  re.split('\?|\.|!', sentences)
                    if 1<len(sentence.strip())] 
                for sentences in full]

    print()
    return full, summ


def remove_samples(data, maxlen_sentence=25, maxlen_word=80):
    gc.collect()
    list_idx = [idx for idx, sentences in enumerate(data[0]) if len(sentences) <= maxlen_sentence]

    list_idx = [idx for idx in list_idx 
                    if max([len(words.split()) for words in data[0][idx]]) <= maxlen_word]

    full = [data[0][idx] for idx in list_idx]
    summ = [data[1][idx] for idx in list_idx]

    return full, summ


def main():
    # Get arguments
    args = parse_arguments()
    
    # Get dataset
    data = read_data(args.input_path)
    
    # Preprocessing data
    processed_data = processing_data(data)
    
    # Remove sample out of maxlen_sentence and maxlen_word
    processed_data = remove_samples(processed_data, args.maxlen_sentence, args.maxlen_word)
    
    # Save processed data
    save_processed(processed_data, args.output_path)


if __name__ == "__main__":
    main()
