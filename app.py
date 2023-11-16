import streamlit as st  
import numpy as np
import joblib,string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


ps = PorterStemmer()
vectorizer=joblib.load('vectorizer.pkl')
model=joblib.load('Model.pkl')
st.title("SMS Spam Predictor")




def transform_text(text) :
    text=text.lower()
    text=text.split()
    y=[]
    for i in text :
        if i.isalnum() :
            y.append(i)
    text=y[:]
    y.clear()
    for i in text :
        if i not in stopwords.words('english') and i not in string.punctuation :
            y.append(i)
    text=y[:]
    y.clear()

    for i in text :
        y.append(ps.stem(i))

    return " ".join(y)





input_text=st.text_input('Enter any message')


if st.button('Predict') :
    input_text=transform_text(input_text)
    x=vectorizer.transform([input_text]).toarray()
    # x=np.expand_dims(x,axis=0)
    pred=model.predict(x)[0]
    if pred :
        st.header('Spam Message')
    else :
        st.header('Not Spam')

