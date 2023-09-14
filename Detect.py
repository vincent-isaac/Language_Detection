import re
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import pickle
from keras.models import load_model

cv=CountVectorizer()
model = load_model('files\LangDetect.h5')
data =input("Check language : ")

sentences = pickle.load(open('files\sentenceses.pkl','rb'))
languages = pickle.load(open('files\Language.pkl','rb'))

def detect(text):
    data_list = []
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', ' ', text)
    text = re.sub(r'[[]]', ' ', text)
    text = text.lower()
    data_list.append(text)

    return data_list

def predict(text):
    sentences.extend(text)
    sentence_vector = cv.fit_transform(sentences).toarray()
    y = np.argmax(model.predict(sentence_vector[-2:]), axis=1)
    output=languages[y[1]]
    return output

process_data=detect(data)
output= predict(process_data)

print(output)