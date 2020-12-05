import gc
import os

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import tokenizer_from_json


def load_tokenizer(path):
    print('Load tokenizer from json...')
    with open(path, "r") as f:
        token = tokenizer_from_json(f.read())
    return token
    

# Create tokenier
def create_tokenizer(samples, filepath_tokenizer, size_vocab=30000, new_token=False):
    gc.collect()

    # Load Tokenizer if tokenizer exist
    if not new_token and os.path.exists(filepath_tokenizer):
        return load_tokenizer(filepath_tokenizer)

    # Create new tokenizer
    print('Create new tokenizer...')
    sentence_list = [sentence 
                        for col in samples
                            for sentences in col
                                for sentence in sentences]
    print('\t- Fit on texts...')
    token = Tokenizer(oov_token='<oov>', num_words=size_vocab+2, filters='')
    token.fit_on_texts(sentence_list)
    
    print(f'\t- Save tokenizer to {filepath_tokenizer}...\n')
    to_json = token.to_json()
    with open(filepath_tokenizer, "w") as f:
        f.write(to_json)

    return token


# Transform text to sequence
def transform_tokenizer(token, samples):
    print('Transform tokenizer...')
    gc.collect()
    samples =  [token.texts_to_sequences(sentences) for sentences in samples]

    return samples
    

def main():
    print('Tokenizer file')
    
    
if __name__ == "__main__":
    main()