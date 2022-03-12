# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 12:35:55 2022

@author: shriy
"""

from flask import Flask,render_template,request,url_for

import pickle


#from final_year_project import TweetClassifier

from final_year_project import process_message



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
        
        
        pm =process_message(message)
        

       
      
        my_prediction=clf.classify(pm)
    return render_template('result.html',prediction=my_prediction)
        
        

if __name__ == '__main__':
    
    
    app.run(debug=False)

 