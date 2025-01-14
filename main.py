import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import re
import streamlit as st

VOCAB_SIZE = 10000

#Load the imdb dataset word index
word_to_index = imdb.get_word_index() # Returns all the word to index dict of vocab

#Reversing the word to index - index to word
index_to_word = {index+3:word for word,index in word_to_index.items()} 

# Load Model file

model = load_model('rnn_imdb.h5')
model.compile(optimizer='adam', 
             loss='binary_crossentropy', 
             metrics=['accuracy'])
model.summary()

# Helper functions
#Function to decode the review
def decode_review(encoded_review):
    """
    Decodes a review from the IMDB dataset using index_to_word mapping.

    Args:
        review_index (int): The index of the review in encoded_review to decode. Defaults to 0.

    Returns:
        str: The decoded review as a string of words.
    """
    return " ".join(index_to_word.get(index, '<UNK>') for index in encoded_review)

#Function to preprocess user input

def preprocess_text(text, max_length=500):
    """
    Preprocesses the user input text, encoding it into indices and padding it to the expected input length.

    Args:
        text (str): The raw text input from the user.
        max_length (int): The maximum length of the input sequence.

    Returns:
        np.array: A numpy array containing the encoded and padded review.
    """

     # Clean the text
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    
    words = text.lower().split()

    encoded_review = []
    for word in words:
        idx = word_to_index.get(word, 2)  # Default to <UNK> token
        if idx + 3 >= VOCAB_SIZE:  # Check if index exceeds vocab size
            idx = 2  # Use <UNK> for words outside vocab size
        encoded_review.append(idx + 3)
    
    # Padding the sequence to ensure consistent input length
    padded_review = sequence.pad_sequences([encoded_review],maxlen=max_length)

    return padded_review


#Streamlit App
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):

    preprocessed_input=preprocess_text(user_input)

    ## Make prediction
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')
