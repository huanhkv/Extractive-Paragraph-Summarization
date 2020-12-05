import os
import sys
path_file = os.path.abspath(__file__)
path_utils = os.path.join(os.path.dirname(os.path.dirname(path_file)), 'utils')
sys.path.insert(0, path_utils)

import tensorflow as tf

from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import Constant
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Embedding, Conv2D, AveragePooling2D, Reshape,
                                     GlobalAveragePooling1D, Dropout, Dense, Concatenate)

from tqdm import tqdm
import numpy as np
import pandas as pd

import gc
import time
import argparse

from utils import MyGenerator, get_maxlen
from tokenizer import load_tokenizer
from io_dataset import read_tokenized


def parse_arguments():
    parser = argparse.ArgumentParser(description='Making the dataset.')
    parser.add_argument('--train_folder', type=str, required=True, help='Path to folder training set')
    parser.add_argument('--valid_folder', type=str, help='Path to folder validation set')
    parser.add_argument('--epochs', type=int, required=True, help='This is the number epochs to train')
    parser.add_argument('--path_tokenizer', type=str, required=True, help='Path to saved tokenizer')
    parser.add_argument('--output_model', type=str, required=True, help='Path to save model')
    
    parser.add_argument('--use_tpu', type=bool, default=True, help='Path to save model')
    parser.add_argument('--filepath_logger', type=str, default='models/log.csv', help='Path to save model')
    parser.add_argument('--filepath_embedding', type=str, required=True, help='')
    parser.add_argument('--filepath_model_minloss', type=str, default='models/model_minloss.h5', help='Path to save model')
    
    return parser.parse_args()

def detect_tpu():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print('Connected!')
        return True
    except:
        print('Not connected to a TPU runtime!')
        return False

def create_environment():
    seed_value= 1907
    os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
    os.environ['PYTHONHASHSEED']=str(seed_value)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    print('Tensorflow Version:', tf.__version__)
    
    tpu_activate = detect_tpu()
    return tpu_activate


def get_embedding_matrix(filepath, token):
    # Read from file to dict
    embedding_dict={}
    with open(filepath, 'r', encoding="utf8") as f:
        for line in f:
            values=line.split()
            word=values[0]
            vectors=np.asarray(values[1:],'float32')
            embedding_dict[word]=vectors
    
    # Get vector
    size_vocab = token.num_words - 2
    embedding_matrix=np.zeros((size_vocab + 2, 100))
    word_index = token.word_index
    
    for word,i in tqdm(word_index.items()):
        if i > size_vocab + 1:
            continue
            
        emb_vec=embedding_dict.get(word)
        if emb_vec is not None:
            embedding_matrix[i]=emb_vec

    return embedding_matrix


def build_model(input_shape=None, 
                output_units=None, 
                size_vocab=None, 
                embedding_matrix=None,
                dropout=None,
                image_model=False):

    print('Create new model...')
    inputs = Input(input_shape)

    if embedding_matrix is not None:
        embedding = Embedding(size_vocab + 2, 100, name='Embedding',
                              trainable=False, 
                              embeddings_initializer=Constant(embedding_matrix))(inputs)
    else:
        embedding = Embedding(size_vocab + 2, 100, name='Embedding')(inputs)
    
    # Block convolution
    block1_conv = Conv2D(100, (1, 3), padding="same", activation="relu", name='block1_conv1')(embedding)
    block1_conv = Conv2D(100, (1, 3), padding="same", activation="relu", name='block1_conv2')(block1_conv)
    block1_pool = AveragePooling2D((1,2), name='block1_pool')(block1_conv)

    block2_conv = Conv2D(200, (1, 3), padding="same", activation="relu", name='block2_conv1')(block1_pool)
    block2_conv = Conv2D(200, (1, 3), padding="same", activation="relu", name='block2_conv2')(block2_conv)
    block2_pool = AveragePooling2D((1,2), name='block2_pool')(block2_conv)

    conv = Conv2D(400, (1, 3), padding="same", activation="relu", name='conv')(block2_pool)
    pool = AveragePooling2D((1,conv.shape[2]), name='pool')(conv)
    reshape = Reshape((pool.shape[1], pool.shape[3]), name='reshape')(pool)

    # Get feature for document
    x1 = GlobalAveragePooling1D(data_format='channels_last', name='document1')(reshape)
    x1 = Dropout(dropout, name='document2')(x1)
    x1 = Dense(x1.shape[1], activation='relu', name='document3')(x1)

    # Get feature for each sentences
    x2 = GlobalAveragePooling1D(data_format='channels_first', name='sentence1')(reshape)
    x2 = Dropout(dropout, name='sentence2')(x2)
    x2 = Dense(25, activation='relu', name='sentence3')(x2)

    # Concatination features of document and features of each sentence
    concat = Concatenate()([x1, x2])
    x = Dense(100, activation='relu')(concat)
    x = Dropout(dropout)(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(dropout)(x)
    outputs = Dense(output_units, activation='softmax')(x)

    model = Model(inputs, outputs)
    
    if image_model:
        plot_model(model, os.path.join(args.output_model, 'plot_model.png'), show_shapes=True)
    return model
    

def train_model(model, x, y=None, 
                optimizer='adam', loss=None, metrics=None,
                epochs=1, 
                validation_data=None, 
                callbacks=None,
                tpu_activate=False,
                filepath_logger=None,
                filepath_model = None,
                filepath_model_minloss = None,
                replace_logger=True):

    gc.collect()
    
    if replace_logger:
        os.remove(filepath_logger)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # If not fit with Generator
    if type(x) is not MyGenerator:
        print(type(x), type(x) is not MyGenerator)
        model.fit(x, y, 
                  epochs=epochs, 
                  validation_data=validation_data,
                  callbacks=callbacks)
        
    # If use TPU fit with Generator
    elif tpu_activate:
        print('Use TPU')
        MAX_LOSS = 999999999
        min_loss_val = MAX_LOSS

        for i in range(epochs):
            gc.collect()
            start = time.time()
            print(f'Epoch {i+1}/{epochs}:')
            
            train_return = [0 for metric in model.metrics_names]
            valid_return = [0 for metric in model.metrics_names]

            # Train on training set
            for (x_batch, y_batch) in tqdm(x):
                train_tmp = model.train_on_batch(x_batch, y_batch)
                train_return = [train_return[i] + train_tmp[i] for i in range(len(train_return))]
            print()
            
            # Evaluate on Valid
            for (x_batch, y_batch) in tqdm(validation_data):
                valid_tmp = model.evaluate(x_batch, y_batch, verbose=0)
                valid_return = [valid_return[i] + valid_tmp[i] for i in range(len(valid_return))]
            print()
            
            train_return = [train_return[i]/len(x) for i in range(len(train_return))]
            valid_return = [valid_return[i]/len(validation_data) for i in range(len(valid_return))]
            
            
            print('\t',end='')
            # Show result on each epochs
            for metric, value in zip(model.metrics_names, train_return):
                print(f' - {metric}: {int(value*10000) / 10000}', end='')
                 
            for metric, value in zip(model.metrics_names, valid_return):
                print(f' - {metric}_val: {int(value*10000) / 10000}', end='')
            print()

            ## Check min loss valid
            loss_val = valid_return[0]
            if filepath_model_minloss and loss_val < min_loss_val:
                print(f'\t - val_loss improved from {min_loss_val} to {loss_val}, saving model to {filepath_model_minloss}...')
                min_loss_val = loss_val
                model.save(filepath_model_minloss)
            else:
                print(f'\t - val_loss did not improve from {min_loss_val}')
            
            ## Save model
            if filepath_model:
                model.save(filepath_model)

            if filepath_logger:
                # Check exist file log
                if os.path.exists(filepath_logger):
                    log_csv = pd.read_csv(filepath_logger)
                else:
                    print(f'\t - No such file or directory: {filepath_logger}. Create {filepath_logger}...')
                    name_columns = [metric for metric in model.metrics_names]
                    name_columns += [metric+'_val' for metric in model.metrics_names]
                    log_csv = pd.DataFrame({col:[] for col in name_columns})
                
                new_row = train_return + valid_return
                log_csv.loc[log_csv.shape[0]] = new_row
                
                # Save
                log_csv.to_csv(filepath_logger, index=False)      

            print(f'\n\tTotal time: {time.time() - start}s\n', '='*80, '\n')
            
    # If not use TPU and fit with Generator
    else:
        print("Not use TPU")
        model.fit(x, 
                  epochs=epochs, 
                  validation_data=validation_data,
                  callbacks=callbacks)
    return model

def main():
    print('Train model file...')
    
    # Get arguments
    args = parse_arguments()

    # Check TPU
    if args.use_tpu:
        tpu_activate = create_environment()
    else:
        tpu_activate = False

    tpu_activate = True

    # Get tokenizer
    token = load_tokenizer(args.path_tokenizer)
    
    # Read tokenized
    x_train, y_train = read_tokenized(args.train_folder)
    
    maxlen_sentence, maxlen_word = get_maxlen(x_train)

    # Create generator
    train_generator = MyGenerator(x_train, y_train, 
                                  maxlen_sentence, maxlen_word, 
                                  batch_size=32)

    # Prepare validation set
    if args.valid_folder:
        x_valid, y_valid = read_tokenized(args.valid_folder)
        valid_generator = MyGenerator(x_valid, y_valid, 
                                      maxlen_sentence, maxlen_word, 
                                      batch_size=32)
    
    # Create model
    input_shape = (maxlen_sentence, maxlen_word,)
    output_units = maxlen_sentence
    size_vocab = token.num_words - 2
    dropout = 0.5
    
    embedding_matrix = get_embedding_matrix(args.filepath_embedding, token)
    
    model = build_model(input_shape, output_units, size_vocab, embedding_matrix, dropout, image_model=False)
    model.summary()
    
    model = train_model(model, train_generator,
                        optimizer='adam', loss='mse', metrics=['mse'],
                        epochs = args.epochs,
                        validation_data=valid_generator,
                        filepath_logger=args.filepath_logger,
                        tpu_activate=tpu_activate,
                        filepath_model=args.output_model,
                        filepath_model_minloss=args.filepath_model_minloss)

if __name__ == "__main__":
    main()
    