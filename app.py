#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from flask import Flask, request
from nltk.tokenize import sent_tokenize, word_tokenize
#import spacy
#from spacy import displacy
from collections import Counter
#import en_core_web_sm
#nlp = en_core_web_sm.load()

import numpy as np
from flask import jsonify
from flask import request
import _pickle as cPickle
import pandas as pd

#from nltk.tokenize import sent_tokenize, word_tokenize
from tablib import Dataset
import json
import os
import numpy as np
import re
import random
import csv

#import nltk
#nltk.download('stopwords')
#from nltk.corpus import stopwords

import tensorflow as tf
import tensorflow_hub as hub
import keras
from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Dense, Lambda, Dropout, BatchNormalization




app = Flask(__name__)
@app.route("/")
def post_route():


    #file= request.files['file'].read()
    #dataset = Dataset().load(file)
    #data=exceltodict(dataset)
    #formatoEYcomp=pd.read_excel('formatoEYcomp.xlsx')
    #print(formatoEYcomp.head())
    return "hola"

#app.run(host='127.0.0.1', port=8080, debug=True)
