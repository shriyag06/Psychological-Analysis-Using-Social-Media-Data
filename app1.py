# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 17:59:03 2022

@author: shriy
"""


from flask import Flask,render_template,request,url_for

import pickle
import nltk
nltk.download("punkt")

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
#import matplotlib.pyplot as plt
#from wordcloud import WordCloud
from math import log
import pandas as pd
import numpy as np



#from final_year_project import TweetClassifier

#from final_year_project import process_message



# load the model from disk
filename = 'model.pkl'
clf = pickle.load(open(filename, 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
	return render_template('website.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    
    
    if request.method == 'POST':
        
        message = request.form['message']
        def process_message(message, lower_case = True, stem = True, stop_words = True, gram = 2):
            if lower_case:
                message = message.lower()
            words = word_tokenize(message)
            words = [w for w in words if len(w) > 2]
            if gram > 1:
                w = []
                for i in range(len(words) - gram + 1):
                    w += [' '.join(words[i:i + gram])]
                return w
            if stop_words:
                sw = stopwords.words('english')
                words = [word for word in words if word not in sw]
            if stem:
                stemmer = PorterStemmer()
                words = [stemmer.stem(word) for word in words]   
            return words
        pm =process_message(message)
        my_prediction=clf.classify(pm)
    return render_template('result.html',prediction=my_prediction)
        
        

if __name__ == '__main__':
    
    
    app1.run(debug=True)
