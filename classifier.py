"""
This script classifies text based on the 
provided dataset and number of authors.

Usage:
    python3 classifier.py \
        --number_of_authors <NUM_AUTHORS>\
        --dataset <DATASET_TYPE> \
        --train_test_data_path <PATH_TO_DATA> \
        --glove_model_path <PATH_TO_GLOVE_MODEL>

Sample run command:
    python3 classifier.py \
        --number_of_authors 5 \
        --dataset blogs \
        --train_test_data_path ../train_test_data \
        --glove_model_path ../../glove.840B.300d.zip
"""

# Imports required libraries.
import argparse
import os
import pickle
import urllib.request
import zipfile
from typing import Dict, List
import numpy as np
from colorama import Fore
from keras.layers import Conv1D, Dense, Dropout, Embedding, Flatten, MaxPooling1D, concatenate, Input, Lambda
from keras.models import Model, Sequential
from keras.preprocessing import text
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import keras.backend as K

# Downloads glove model.
def download_glove(filepath: str):
    print("GloVe model not found. Downloading from http://nlp.stanford.edu/data/glove.840B.300d.zip ...")
    url = "http://nlp.stanford.edu/data/glove.840B.300d.zip"
    urllib.request.urlretrieve(url, filepath)

# Loads glove model.
def load_model(filepath: str) -> dict:
    parent_dir = os.path.dirname(filepath)
    print(f"Loading GloVe model from {filepath} ...")
    if not os.path.exists(filepath):
        download_glove(filepath)
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            zip_ref.extractall(parent_dir)
    embeddings_index = {}
    with open(filepath.replace(".zip", ".txt"), "r", encoding="utf-8") as file:
        for line in file:
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype="float32")
                if len(coefs) == 300:
                    embeddings_index[word] = coefs
            except:
                continue
    print(f"Found {len(embeddings_index)} word vectors.")
    return embeddings_index

# Defines a CNN model with two parallel embedding layers (trainable and non-trainable),
# and returns a compiled CNN model.
def model_cnn_word_word(
    embedding_matrix: Dict[str, np.ndarray], # A dictionary of word embeddings
    word_index: Dict[str, int],              # A dictionary of word indices
    embedding_dim: int,                      # The dimension of the word embeddings
    max_sequence_length: int,                # Maximum length of the input sequence
    number_of_authors: int,
) -> Sequential:
    
    print('>>>>>>', max_sequence_length)
    embed1_in  = Input(shape=(None, 12998))
    embed1_in = tf.reshape(embed1_in, shape=[tf.shape(embed1_in)[0]*tf.shape(embed1_in)[1],12998])
    embed1_out = Embedding(
            len(word_index) + 1,
            embedding_dim,
            weights=[embedding_matrix],
            input_length=max_sequence_length,
            trainable=False,
        )(embed1_in)
    #embed1_out = embed1_out[0]
    print('>>>>>> embed1_out:', K.int_shape(embed1_out))
    embed1 = Model(embed1_in, embed1_out)
    embed2_in  = Input(shape=(None, 12998))
    embed2_in = tf.reshape(embed2_in, shape=[tf.shape(embed2_in)[0]*tf.shape(embed2_in)[1],12998])
    embed2_out = Embedding(
            len(word_index) + 1,
            embedding_dim,
            weights=[embedding_matrix],
            input_length=max_sequence_length,
            trainable=True,
        )(embed2_in)
    #embed2_out = embed2_out[0]
    print('>>>>>> embed2_out:', K.int_shape(embed2_out))
    embed2 = Model(embed2_in, embed2_out)
    conc1 = concatenate([embed1_out, embed2_out])
    print('>>>>>> conc1:', K.int_shape(conc1))
    conc2 = Conv1D(64, 5, activation="relu")(conc1)
    print('>>>>>> conc2:', K.int_shape(conc2))
    #conc2 = tf.reshape(conc2, shape=[12998, 296, 64])
    conc3 = MaxPooling1D(max_sequence_length - 5 + 1)(conc2)
    print('>>>>>> conc3:', K.int_shape(conc3))
    conc4 = Flatten()(conc3)
    print('>>>>>> conc4:', K.int_shape(conc4))
    conc5 = Dense(256, activation="relu")(conc4)
    print('>>>>>> conc5:', K.int_shape(conc5))
    conc6 = Dropout(0.5)(conc5)
    print('>>>>>> conc6:', K.int_shape(conc6))
    conc7 = Dense(number_of_authors, activation="softmax")(conc6)
    print('>>>>>> conc7:', K.int_shape(conc7))
    model = Model([embed1_in, embed2_in], conc7)
    
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["acc"])
    plot_model(model, to_file='model.png')
    print(model.summary())
    return model

# Takes in a dictionary of word embeddings and a dictionary of word indices,
# and returns a matrix of word embeddings where any missing words are filled in with zeros.
def fill_in_missing_words_with_zeros(
    embeddings_index: Dict[str, np.ndarray], # A dictionary of word embeddings
    word_index: Dict[str, int],              # A dictionary of word indices
    EMBEDDING_DIM: int,                      # The dimension of the word embeddings
) -> np.ndarray:
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector # words not found in embedding index will be all-zeros
    return embedding_matrix

# Create a tokenizer object and fit it on the given lines of text,
# and returns a tokenizer object that has been fit on the given text.
def create_tokenizer(lines: List[str]) -> Tokenizer: # A list of strings representing the text to fit the tokenizer on
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# Loads data from a pickle file, and returns data loaded from the pickle file.
def load_pickle_data(path: str): # Path to the pickle file
    with open(path, "rb") as file:
        data = pickle.load(file)
    return data

# Cleans the provided text by removing special characters and converting the text to a sequence of words,
# and returns cleaned text string.
def get_clean_text(input_text: str) -> str: # Raw text string to be cleaned
    clean_text_list = text.text_to_word_sequence(
        input_text, filters="", lower=False, split=" "
    )
    clean_text = " ".join(clean_text_list)
    return clean_text

# Extracts the features for inputText and authorLabel from each row,
# as each row contains data in the following format: (input_text, author_label),
# and returns list of tuples containing the cleaned text and the author id.
def prepare_data_for_classification(data: List[tuple]) -> tuple: # List of tuples containing the raw text and the author id
    X_data = []
    y_data = []
    for input_text, author_label in data:
        clean_text = get_clean_text(input_text)
        X_data.append(clean_text)
        y_data.append(author_label)
    return X_data, y_data

# Trains a classifier using the provided training data,
# and returns trained classifier and StandardScaler used to scale the data.
def train_classifier(
    X_train: List[List[float]],  # List of feature vectors for training data
    y_train: List[int],          # List of author ids for training data
    glove_filepath: str,
    number_of_authors: int,      # Number of authors in the dataset
    embedding_dim: int = 300,
) -> tuple:
    tokenizer = create_tokenizer(X_train)
    sequences = tokenizer.texts_to_sequences(X_train)
    max_sequence_length = max([len(article.split()) for article in X_train])
    vocab_size = len(tokenizer.word_index) + 1
    print(f"\nMax document length: {max_sequence_length}")
    print(f"Vocabulary size: {vocab_size}\n")
    x_train = pad_sequences(sequences, maxlen=max_sequence_length)
    y_train = to_categorical(y_train)
    print(f"Shape of data tensor: {x_train.shape}")
    print(f"Shape of label tensor: {y_train.shape}\n")
    glove_embedding_matrix = load_model(glove_filepath)
    print("Filling non existing words\n")
    glove_embedding_matrix = fill_in_missing_words_with_zeros(
        glove_embedding_matrix, tokenizer.word_index, embedding_dim
    )
    model = model_cnn_word_word(
        glove_embedding_matrix,
        tokenizer.word_index,
        embedding_dim,
        max_sequence_length,
        number_of_authors,
    )
    model.fit([x_train, x_train], y_train, epochs=15, batch_size=50)
    return tokenizer, model, max_sequence_length

# Tests the classifier using the provided test data.
def test_classifier(
    X_test: List[List[float]], # List of feature vectors for test data
    y_test: List[int],         # List of author ids for test data
    tokenizer: Tokenizer,      # Tokenizer used to tokenize the data
    max_sequence_length: int,  # Maximum length of the input sequence
    model: Sequential,         # Trained classifier
) -> None:
    sequences = tokenizer.texts_to_sequences(X_test)
    y_test = to_categorical(y_test)
    x_test = pad_sequences(sequences, maxlen=max_sequence_length)
    _, acc = model.evaluate([x_test, x_test], y_test, verbose=0)
    print("Test Accuracy: %f" % (acc * 100))

# Main function to process the dataset, train a classifier, and test it.
def main(
    number_of_authors: int,      # Number of authors in the dataset
    dataset_type: str,           # Type of dataset (e.g., 'amt', 'blogs', etc.)
    train_test_data_path: str,   # Directory path containing training and testing pickle data files
    glove_model_path: str,       # Path to the GloVe model
) -> None:
    print(Fore.WHITE)
    print(f"\nNumber of Authors: {number_of_authors}")
    print(f"Dataset Type: {dataset_type}")
    print(f"Train/Test Data Path: {train_test_data_path}\n")
    train_data_path = (
        f"{train_test_data_path}/{str(number_of_authors)}"
        f"_train_files_{dataset_type}.pkl"
    )
    test_data_path = (
        f"{train_test_data_path}/{str(number_of_authors)}"
        f"_test_files_{dataset_type}.pkl"
    )
    train_data = load_pickle_data(train_data_path)
    test_data = load_pickle_data(test_data_path)
    X_train, y_train = prepare_data_for_classification(train_data)
    X_test, y_test = prepare_data_for_classification(test_data)
    print(f"Total Train Articles: {len(train_data)}")
    print(f"Total Test Articles: {len(test_data)}\n")
    tokenizer, model, max_sequence_length = train_classifier(
        X_train, y_train, glove_model_path, number_of_authors
    )
    test_classifier(X_test, y_test, tokenizer, max_sequence_length, model)
    print(Fore.RESET)

# Main program
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test CNN classifier.")
    parser.add_argument(
        "--number_of_authors",
        type=int,
        help="Number of authors to keep for classification.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["blogs"],
        help='Choose "blogs" dataset.',
    )
    parser.add_argument(
        "--train_test_data_path", type=str, help="Path to the train/test data."
    )
    parser.add_argument(
        "--glove_model_path", type=str, help="Path of glove.840B.300d.zip model."
    )
    args = parser.parse_args()
    main(
        args.number_of_authors,
        args.dataset,
        args.train_test_data_path,
        args.glove_model_path,
    )