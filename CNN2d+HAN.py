#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 15:38:07 2018

@author: sarthakmehta
"""
import random
import os
from keras.preprocessing.text import text_to_word_sequence
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.callbacks import CSVLogger
from collections import Counter
import glob
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D, Flatten
from keras.optimizers import Adam 
import numpy as np 

Train_x= []
Train_y= []
Test_x= []
Test_y= []

path_pos = os.getcwd() +'/aclImdb/train/pos/*.txt'  
path_neg =os.getcwd() +'/aclImdb/train/neg/*.txt'  
path_pos_test = os.getcwd() +'/aclImdb/train/pos/*.txt'  
path_neg_test =os.getcwd() +'/aclImdb/train/neg/*.txt'

def get_class(file):
    out = file.split("_")
    num = out[1].split(".")
    return int(num[0])

def load_data(path,Train_x,Train_y):  
     
    files=glob.glob(path)   
    for file in files: 
        Train_y.append(get_class(file))
        f=open(file, 'r')  
        Train_x.append(f.read())
        f.close()                  

def pre_process(X):
    out=[]
    num_of_word =[]
    num_of_charac = []
    for x in X:
        out.extend(text_to_word_sequence(x))
        num_of_word.append(len(text_to_word_sequence(x)))
    for y in out:
        num_of_charac.append(len(y))
    tokenizer= Tokenizer(char_level=True)
    tokenizer.fit_on_texts(out)
    
    return tokenizer,max(num_of_word),max(num_of_charac)

def encode(X,Y):
    global tokenizer,max_len_word,max_len_char
    while 1: 
        for x in range(0,len(X)):
            en_x = np.zeros((max_len_word,max_len_char,110) )
            words = text_to_word_sequence(X[x])
            en_y = np.zeros(10)
            en_y[Y[x]-1] = 1
            en_y=np.reshape(en_y,(1,10))
            for i in range(0,len(words)):
                for j in range(0,len(words[i])):
                    en_x[i][j][tokenizer.word_index[words[i][j]]] = 1
                   
            en_x=np.reshape(en_x,(1,max_len_word,max_len_char,110))
            yield en_x,en_y
                
            
                     
   

def make_model(input_shape,output_shape):
    model = Sequential()
    model.add(Conv2D(32,
                 (110,3),
                 padding='same',input_shape = (2493,51,input_shape),
                 activation='tanh',
                 strides=(1,1)))
    model.add(Flatten())
    model.add(Dense(output_shape,activation='softmax'))
    model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001),
              metrics=['accuracy'])
    return model


load_data(path_pos,Train_x,Train_y)
load_data(path_neg,Train_x,Train_y)

load_data(path_pos,Test_x,Test_y)
load_data(path_neg,Test_x,Test_y)

permutation = random.sample(range(0,25000), 25000)

Train_x=[Train_x[i] for i in permutation]
Train_y=[Train_y[i] for i in permutation]

tokenizer,max_len_word,max_len_char = pre_process(Train_x)

out=''
csv_logger = CSVLogger('log.csv', append=True, separator=';')
    
inp_shape= 110
out_shape = 10

model = make_model(inp_shape,out_shape)

model.fit_generator(encode(Train_x[0:20000],Train_y[0:20000]), steps_per_epoch=1000, epochs=10,callbacks=[csv_logger],validation_data=encode(Train_x[20000:],Train_y[20000:]),validation_steps=100)
model.save_weights("model.h5")
print("Saved model to disk")







    