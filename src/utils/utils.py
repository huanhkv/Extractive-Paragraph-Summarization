import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence

import os

# Generator data
class MyGenerator(Sequence):
    def __init__(self, x, y, maxlen_sentence, maxlen_word, batch_size=32):
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.maxlen_sentence = maxlen_sentence
        self.maxlen_word = maxlen_word
        
    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch_x = self.x[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_y = self.y[idx*self.batch_size:(idx+1)*self.batch_size]
        
        batch_x = padding_doc(batch_x, self.maxlen_sentence, self.maxlen_word)
        batch_y = pad_sequences(batch_y, self.maxlen_sentence, dtype='float32')

        return batch_x, batch_y


def padding_doc(samples, maxlen_sentence, maxlen_word):
    def pad_sentences(sentences, maxlen_sentence, maxlen_word):
        len_add = maxlen_sentence - sentences.shape[0]
        zeros = np.zeros((len_add, maxlen_word))
        return np.concatenate((sentences, zeros))

    samples = np.array([pad_sequences(sentences, maxlen=maxlen_word, padding='post')
                            for sentences in samples])

    samples = np.array([pad_sentences(sentences, maxlen_sentence, maxlen_word) 
                            for sentences in samples])
    
    return samples


def get_maxlen(samples):
    maxlen_sentence = max([len(sentences) for sentences in samples])
    maxlen_word = max([len(words) for sentences in samples for words in sentences])
     
    return maxlen_sentence, maxlen_word
