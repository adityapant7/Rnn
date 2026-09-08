import tensorflow as tf 
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import streamlit as st
model=load_model('model.h5')
word_index=imdb.get_word_index()
reverse_index={value:key for key,value in word_index.items()}
# function to decode
def decode(encode):
    return ' '.join([reverse_index.get(i-3,'?') for i in encode])
#preprocessing of new input
def preprocess_text(text):
    words=text.lower().split()
    encode=[word_index.get(i,2)+3 for i in words]
    pad_review=sequence.pad_sequences([encode],maxlen=500)
    return pad_review
def predict_sentiment(review):
    predict_sentiment_output=preprocess_text(review)
    prediction=model.predict(predict_sentiment_output)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'
    return sentiment,prediction[0][0]
#streamlit app
st.title("IMDB SENTIMENT ANALYSIS")
st.write("Enter a movie review to check if it is positive or negative ")
user_review=st.text_area("enter your review")
if st.button('classification'):
    sentiment,score=predict_sentiment(user_review)
    st.write("sentiment=",sentiment)
    st.write("prediction score=",score)
else:
    st.write("enter review")


