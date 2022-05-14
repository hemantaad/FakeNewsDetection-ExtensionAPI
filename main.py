from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import string
import json

string.punctuation
import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.models import load_model

reloadModel = load_model('lstmmodel.h5')

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route('/predict', methods = ['POST'])
def news_url():
    news = request.json
    news = news[0]
    testData = pd.DataFrame({'x': news}).transpose()
    print(testData)
    testData.fillna('unavailable', inplace=True)
    testData['comb'] = testData['head'] + "_" + testData['body'] + "_" + testData['author']
    wordnet = WordNetLemmatizer()
    stemmer = PorterStemmer()

    def clean(text):
        text = "".join([re.sub('[^a-zA-Z]', ' ', char) for char in text])
        text = text.lower()
        text = text.split()
        text = [stemmer.stem(word) for word in text if word not in set(stopwords.words("english"))]
        text = " ".join(text)
        return text

    testData['comb'] = testData['comb'].apply(clean)
    vocab_size = 10000
    news_title = testData['comb']
    one_hot_r = [one_hot(words, vocab_size) for words in news_title]
    sent_len = 800
    final_input = pad_sequences(one_hot_r, padding='post', maxlen=sent_len)
    result = reloadModel.predict(final_input)[0]
    final_result = result[0]
    final_result = str(final_result)
    return jsonify({'final_result' : final_result})


if __name__ == '__main__':
    app.run(debug=True)
