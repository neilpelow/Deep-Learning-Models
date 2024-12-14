from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import os

def load_training_data(data_dir):
  # Load the training data (assuming it's a CSV file)
  training_data_path = os.path.join(data_dir, 'train.csv')
  training_data = pd.read_csv(training_data_path)
    
  # Return the training data
  return training_data

def prepare_training_data(training_data, num_words=10000, oov_token='<OOV>', maxlen=100, test_size=0.2, random_state=42):
    """
    Prepares data for training deep learning models by tokenizing lyrics, encoding genres, and splitting into training and validation sets.

    Args:
        training_data (pd.DataFrame): A DataFrame with 'Lyrics' and 'Genre' columns.
        num_words (int): Maximum number of words to keep in the tokenizer. Default is 10,000.
        oov_token (str): Token for out-of-vocabulary words. Default is '<OOV>'.
        maxlen (int): Maximum length of sequences for padding. Default is 100.
        test_size (float): Proportion of the data to be used for validation. Default is 0.2.
        random_state (int): Random state for reproducibility. Default is 42.

    Returns:
        tuple: (X_train, X_val, y_train, y_val, tokenizer, label_encoder)
            - X_train: Padded sequences for training.
            - X_val: Padded sequences for validation.
            - y_train: Encoded labels for training.
            - y_val: Encoded labels for validation.
            - tokenizer: Fitted tokenizer instance.
            - label_encoder: Fitted label encoder instance.
    """
    # Prepare the data
    lyrics = training_data['Lyrics'].astype(str).values
    genres = training_data['Genre'].values

    # Tokenize the lyrics
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(lyrics)
    sequences = tokenizer.texts_to_sequences(lyrics)
    padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')

    # Encode the labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(genres)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(padded_sequences, encoded_labels, test_size=test_size, random_state=random_state)

    return X_train, X_val, y_train, y_val, tokenizer, label_encoder
