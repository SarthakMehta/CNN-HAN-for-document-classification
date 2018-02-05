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
#from collections import Counter
import glob
#from keras.models import Sequential
from keras.layers import Dense,Input
from keras.layers import Conv2D, Flatten
from keras.optimizers import Adam 
import numpy as np 
#import threading
#from matplotlib.pyplot import scatter 
#from matplotlib.pyplot import xlabel
#
#from matplotlib.pyplot import ylabel
Train_x= []
Train_y= []
Test_x= []
Test_y= []

path_pos = os.getcwd() +'/aclImdb/train/pos/*.txt'  
path_neg =os.getcwd() +'/aclImdb/train/neg/*.txt'  
path_pos_test = os.getcwd() +'/aclImdb/train/pos/*.txt'  
path_neg_test =os.getcwd() +'/aclImdb/train/neg/*.txt'

#class myThread (threading.Thread):
#   def __init__(self, threadID, name):
#      threading.Thread.__init__(self)
#      self.threadID = threadID
#      self.name = name
#      
#   def run(self):
#      print("Starting " + self.name)
#      get_gusage(self.name)
#      print("Exiting " + self.name)

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

#def get_gusage( threadName):
#    print(threadName)
#    os.system('nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 5')             

load_data(path_pos,Train_x,Train_y)
load_data(path_neg,Train_x,Train_y)

load_data(path_pos,Test_x,Test_y)
load_data(path_neg,Test_x,Test_y)

permutation = random.sample(range(0,25000), 25000)

Train_x=[Train_x[i] for i in permutation]
Train_y=[Train_y[i] for i in permutation]

tokenizer,max_len_word,max_len_char = pre_process(Train_x)

out=''
csv_logger = CSVLogger('log2.csv', append=True, separator=';')
    
inp_shape= 110
out_shape = 10

#MN=[]
#
#for i in range(1,1024):
#    
##########################################################################################

in_1 = Input(shape=(2493,51,inp_shape))

conv2D_2 = Conv2D(300,
                 (3,110),
                 padding='same',
                 activation='tanh',
                 strides=(1,110),data_format="channels_first")(in_1)

flat_3 = Flatten()(conv2D_2)

dense_4 = Dense(out_shape,activation='softmax')(flat_3)


##########################################################################################


model = Model(in_1,dense_4)
    
#    MN.append(get_model_memory_usage(1,model))


model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001),
              metrics=['accuracy'])



#scatter(range(1,1024),MN)
#xlabel("number of convolution filters")
#ylabel("Memory need in Gigabytes")
#thread1 = myThread(1, "Thread-1")
#thread1.start()

model.fit_generator(encode(Train_x[0:20000],Train_y[0:20000]), steps_per_epoch=25000, epochs=10,callbacks=[csv_logger],validation_data=encode(Train_x[20000:22500],Train_y[20000:22500]),validation_steps=2500)
score = model.evaluate_generator(encode(Train_x[22500:],Train_y[22500:]),steps=2500,max_queue_size=2500)
model.save_weights("model.h5")
print('Test score:', score[0])
print('Test accuracy:', score[1])
print("Saved model to disk")







    