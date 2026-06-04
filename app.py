import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import numpy as np
import pickle
import tf_keras
from tf_keras.preprocessing.sequence import pad_sequences

#Load the LSTM Model
model = tf_keras.models.load_model('next_word_lstm.h5')

#3 Laod the tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]  # Convert the input text to a sequence of integers
    if len(token_list) >= max_sequence_len:             # If the input sequence is longer than max_sequence_len-1, truncate it
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre') # Pad the sequence to the required length
    predicted = model.predict(token_list, verbose=0) # Predict the next word using the model
    predicted_word_index = np.argmax(predicted, axis=1) # Get the index of the predicted word
    for word, index in tokenizer.word_index.items(): # Iterate through the tokenizer's word index to find the word corresponding to the predicted index
        if index == predicted_word_index:
            return word
    return None

# streamlit app
st.title("Next Word Prediction With LSTM And Early Stopping")
input_text=st.text_input("Enter the sequence of Words","To be or not to")
if st.button("Predict Next Word"):
    max_sequence_len = model.layers[0].input_length + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f'Next word: {next_word}')

