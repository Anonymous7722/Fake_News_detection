import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import re


lr=pickle.load(open('lr.pkl','rb'))
vect=pickle.load(open('vect.pkl','rb'))

port_stem = PorterStemmer()
def Fake_news(news):
    te = re.sub('[^a-zA-Z]',' ',news)
    te = te.lower()
    te = te.split()
    te = [port_stem.stem(word) for word in te if not word in stopwords.words('english')]
    te = ' '.join(te)
    te=[te]
    te = vect.transform(te).toarray()
    te=lr.predict(te)
    print(te)
    if te[0] == 0:
        te='Fake News Detected!'
    else:
        te='This is Not a Fake News'
    return te



st.title('Fake News Detection')
st.write('This is a simple web app that uses a machine learning model to detect fake news')


# prompt = st.chat_input("Say something")
# if prompt:
#     st.write(f"User has sent the following prompt: {prompt}")
messages = st.container(height=400)
messages.chat_message("assistant").write("Model: Enter any news to detect wether it is Fake or Not")
if prompt := st.chat_input("Say something"):
    messages.chat_message("user").write(f"User: {prompt}")
    news=Fake_news(prompt)
    messages.chat_message("assistant").write(f"Model: {news}")

st.caption('Made by Shaniyaz')