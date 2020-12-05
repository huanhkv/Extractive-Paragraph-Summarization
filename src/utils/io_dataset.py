import gc
import os
import numpy as np


def read_from_txt(path):
    print('Read data from txt...')
    gc.collect()
    with open(path, 'r', encoding="utf-8") as f:
        docs = f.read().split('\n')
    return docs
    
    
def save_processed(processed_data, path):
    print('Save processed data...')
    # Write full text to file
    with open(os.path.join(path, 'processed_full.txt'), 'w') as processed_full:
        # Join list to string
        full_txt = ['.'.join(sample) for sample in processed_data[0]]
        full_txt = '\n'.join(full_txt)
        
        processed_full.write(full_txt)
    
    # Write summary text to file
    with open(os.path.join(path, 'processed_summ.txt'), 'w') as processed_summ:
        # Join list to string
        summ_txt = '\n'.join(processed_data[1])
        
        # Write summ text to file
        processed_summ.write(summ_txt)
        

def read_processed(path):
    print('Read processed data...')
    with open(os.path.join(path, 'processed_full.txt')) as f:
        full = f.read().split('\n')
    
    with open(os.path.join(path, 'processed_summ.txt')) as f:
        summ = f.read().split('\n')
    
    full = [sentences.split('.') for sentences in full]
    
    return full, summ


def save_tokenized(path, x, y):
    print('Save tokenized data...')
    # Save x and y
    with open(os.path.join(path, 'x.npy'), 'wb') as f:
        np.save(f, np.array(x, dtype=object))

    with open(os.path.join(path, 'y.npy'), 'wb') as f:
        np.save(f, y)


def read_tokenized(path):
    print('Read tokenized data...')
    with open(os.path.join(path, 'x.npy'), 'rb') as f:
        x = np.load(f, allow_pickle=True)

    with open(os.path.join(path, 'y.npy'), 'rb') as f:
        y = np.load(f)
    
    x = [i for i in x]
    return x, y


def main():
    print('IO Dataset')

if __name__ == '__main__':
    main()