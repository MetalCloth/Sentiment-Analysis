import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.preprocessing import sequence

word_index=imdb.get_word_index()
reverse_word_index=dict([(value,key) for (key,value) in word_index.items()])

file=load_model('p.h5')


def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i,'?') for i in encoded_review])

def process_review(review):
    words=review.lower().split()
    encoded_review=[word_index.get(word,2) +3 for word in words]
    padded_review=pad_sequences([encoded_review],value=0,padding='pre',maxlen=500)
    return padded_review


def predict_sentiment(review):
    preprocessed_review=process_review(review)
    prediction=file.predict(preprocessed_review)
    sentiment='positive' if prediction[0][0]>0.5 else 'negative'
    return sentiment,prediction[0][0]

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review and let's see if it is positive or negative")

user=st.text_area("Moview Review")

if st.button("Predict"):
    sentiment,prediction=predict_sentiment(user)
    st.write(f"Sentiment: {sentiment}")
else:
    st.write("Please enter a movie review")
