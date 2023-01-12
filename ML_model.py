#!/usr/bin/env python
# coding: utf-8

# In[159]:


from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import numpy as np
import pandas as pd
import random
import sys
import io
import os
import urllib


# In[160]:


pwd


# In[161]:


cd Desktop


# In[162]:


cd dr_reddy


# In[163]:


df = pd.read_excel("dataset.xlsx")


# In[164]:


df.head()


# In[165]:


text = df["Drug name"].drop_duplicates()


# In[166]:


text = (list(set(text)))
for i in range(len(text)):
    text[i]=str(text[i])
    text[i]=text[i].lower()


# In[167]:


text = list(set("".join(text)))


# In[168]:


chars = sorted(text)
print('total chars:', len(chars))
print(chars)


# In[169]:


chars.append("/n")


# In[170]:


chars.append("\n")


# In[171]:


# for i in range(26):
#     chars.append(chr(ord('a') + i))


# In[172]:


char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print(char_indices)


# In[173]:


df_drug_names = df["Drug name"].drop_duplicates()


# In[174]:


df_drug_names.head()


# In[175]:


something = df_drug_names.to_list()
for j in range(len(something)):
    something[j] = str(something[j])
    something[j] = something[j].lower()
    something[j] = something[j].split(" ")
    something[j] = something[j][0]
something = set(something)


# In[176]:


something


# In[177]:


lines = something
lines = [line for line in lines if len(line)!=0]


# In[178]:


lines


# In[180]:


maxlen = len(max(lines, key=len)) + 40
minlen = len(min(lines, key=len))

print("line with longest length: "+ str(maxlen))
print("line with shorter length: "+ str(minlen))


# In[181]:


steps = 1
sequences = []
next_chars = []

for line in lines:
    
    s = (maxlen - len(line))*'0' + line
    sequences.append(s)
    next_chars.append('\n')
    for it,j in enumerate(line):
        if (it >= len(line)-1):
            continue
        s = (maxlen - len(line[:-1-it]))*'0' + line[:-1-it]
        sequences.append(s)
        next_chars.append(line[-1-it])


# In[182]:


print('total sequences:', len(sequences))
print(sequences[66], next_chars[66])
print(sequences[67], next_chars[67])
print(sequences[68], next_chars[68])


# In[183]:


x = np.zeros((len(sequences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sequences), len(chars)), dtype=np.bool)
for i, seq in enumerate(sequences):
    for t, char in enumerate(seq):
        if char != '0':
            x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# In[184]:


prefix = ""
max_names = 10

def sample(preds):
    """ function that sample an index from a probability array """
    preds = np.asarray(preds).astype('float64')
    preds = preds / np.sum(preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.random.choice(range(len(chars)), p = probas.ravel())

def print_name_generated(name):
    print(name, flush=True)
def print_list_generated(lst):
    print(lst, flush=True)
    
    
def generate_new_names(*args):
    print("----------Generatinig names----------")

    
    sequence = ('{0:0>' + str(maxlen) + '}').format(prefix).lower()

    
    tmp_generated = prefix
    list_outputs = list()

    while (len(list_outputs) < max_names):

        
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sequence):
            if char != '0':
                x_pred[0, t, char_indices[char]] = 1

        
        preds = model.predict(x_pred, verbose=0)[0]

        
        next_index = sample(preds)
       
        next_char = indices_char[next_index]

        
        if next_char == '\n' or len(tmp_generated) > maxlen:
            
            
            if tmp_generated not in list_outputs:
                list_outputs.append(tmp_generated)
                print_name_generated(tmp_generated)
            
            sequence = ('{0:0>' + str(maxlen) + '}').format(prefix).lower()
            tmp_generated = prefix
        else:
    
            
            tmp_generated += next_char
            
            sequence = ('{0:0>' + str(maxlen) + '}').format(tmp_generated).lower()
            
     
    print("Set of words already in the dataset:")
    print_list_generated(set(lines).intersection(list_outputs))
    
    
    total_repited = len(set(lines).intersection(list_outputs))
    total = len(list_outputs)
    print("Rate of total invented words: " + "{:.2f}".format((total-total_repited)/total))
    print("-----------------End-----------------")
    

callback = LambdaCallback(on_epoch_end=generate_new_names)


# In[185]:


model = Sequential()
model.add(LSTM(64, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))
history = model.fit(x, y, batch_size=128, epochs=2, verbose=2, callbacks=[callback])


# In[ ]:





# In[ ]:


dataset_url = urllib.request.urlopen('https://ipindiaonline.gov.in/tmrpublicsearch/frmmain.aspx')

tf.keras.utils.get_file('dataset.xlsx', dataset_url,
                        extract=True, cache_dir='.')
dataframe = pd.read_xlsx(xlsx_file)


# In[ ]:


def soundex_generator(token):
   
    token = token.upper()
 
    soundex = ""
 
    soundex += token[0]


# In[ ]:


def soundex_generator(token):
   
    token = token.upper()
 
    soundex = ""
 
    soundex += token[0]
 
    dictionary = {history}
 
    for char in token[1:]:
        for key in dictionary.keys():
            if char in key:
                code = dictionary[key]
                if code != '.':
                    if code != soundex[-1]:
                        soundex += code
 
    soundex = soundex[:7].ljust(7, "0")
 
    return soundex


# In[ ]:





# In[ ]:





# In[ ]:




