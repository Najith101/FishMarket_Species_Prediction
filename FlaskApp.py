#!/usr/bin/env python
# coding: utf-8

# In[4]:


from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import pandas as pd


# In[5]:


app=Flask(__name__)
pickle_in=open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)


# In[6]:



@app.route('/')
def home():
    return render_template('index.html')


# In[7]:


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = classifier.predict(final_features)

    
    return render_template('index.html', prediction_text='The fish belong to species {}'.format(prediction))


# In[8]:


if __name__=="__main__":
    app.run()


# In[ ]:




