import os
import sys
path_file = os.path.abspath(__file__)
path_utils = os.path.join(os.path.dirname(os.path.dirname(path_file)), 'utils')
sys.path.insert(0, path_utils)

from rouge import cal_rouge
from io_dataset import read_processed, save_tokenized
from tokenizer import create_tokenizer, transform_tokenizer

import numpy as np

import gc
import json
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Making the dataset.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to input')
    parser.add_argument('--output_path', type=str, required=True, help='Path to output')
    parser.add_argument('--save_tokenizer', type=str, required=True, help='Path to input')
    parser.add_argument('--new_token', default=False, type=bool, help='Path to input')
    
    return parser.parse_args()

def main():
    # Get arguments
    args = parse_arguments()
    
    # Get processed data
    data = read_processed(args.input_path)
    
    # Get tokenizer
    token = create_tokenizer(data, args.save_tokenizer, args.new_token)
    
    # Transform
    x = transform_tokenizer(token, data[0])
    
    # Calculater Rouge
    y = cal_rouge(data)
    
    # Save
    save_tokenized(args.output_path, x, y)


if __name__ == "__main__":
    main()