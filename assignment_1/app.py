#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


# In[4]:


from tensorflow.keras.models import load_model
model = load_model('next_word_predictor.h5')
tokenizer = Tokenizer()
text = open('Data.txt',encoding = "utf-8").read().lower()
tokenizer.fit_on_texts([text])
max_sequence_len = 1233
def predict_next_words(model, tokenizer, text, num_words):
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        pos = np.argmax(model.predict(padded_token_text))

        for word,index in tokenizer.word_index.items():
            if index == pos:
                text = text + " " + word
            return text


# In[5]:


import streamlit as st
st.title('Next Word Prediction')
input_text = st.text_input('Enter your text here')
num_words = st.number_input('Number of words to predict', min_value=1, max_value=50, value=1)

if st.button('Predict'):
    if input_text:
        result = predict_next_words(model, tokenizer, input_text, num_words)
        st.write(result)
    else:
        st.write("Please enter some text")


# In[ ]:




