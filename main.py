import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.lm.preprocessing import pad_sequence
stemmer = SnowballStemmer("english")
from nltk.tokenize import TweetTokenizer
tt = TweetTokenizer()
import numpy as np
import tensorflow as tf # v2.2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
import random
import json
import pandas as pd
import string
import os as os
nltk.download('punkt')

with open(os.path.join(os.getcwd(), 'intents.json')) as json_file:
    skills_df = pd.read_json(json_file)
    skills_df = pd.DataFrame(skills_df.intents.values.tolist())

queries = skills_df.explode('queries').reset_index().drop(['responses', 'index'], axis = 1)
responses = skills_df.explode('responses').reset_index().drop(['queries', 'index'], axis = 1)

def clean_query(query):
    words = nltk.word_tokenize(query)
    words_token = [stemmer.stem(word.lower()) for word in words if word.isalpha()]
    return words_token

queries['queries'] = queries['queries'].astype('str')
queries['queries_token'] = queries['queries'].apply(clean_query)

def create_vocab(queries):
    vocab = []
    for query in queries:
        vocab.extend(query)
    return list(set(vocab))

intents = list(skills_df['intent'])
vocab = create_vocab(queries['queries_token'])

def bag_words(query):
    bow = [0] * len(vocab)
    for word in query:
        index = vocab.index(word) if word in vocab else None
        if index == None: continue
        bow[index] += 1
    return bow

def intent_no(intent):
    intent_arr = [0] * len(intents)
    intent_arr[intents.index(intent)] = 1
    return intent_arr

queries['bow'] = queries['queries_token'].apply(bag_words)
queries['class'] = queries['intent'].apply(intent_no)

model = Sequential()
model.add(Dense(256, input_shape=(len(queries['bow'][0]),), activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(len(queries['class'][0]), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(np.array(queries['bow'].tolist()), np.array(queries['class'].tolist()), epochs=200, batch_size=5, verbose=1)

def classify_query(query):
    confidence = 0.6
    
    query = clean_query(query)
    query_bag = bag_words(query)
    predictions = model.predict(np.array([query_bag]))[0]
    
    choosen_class = -1 # Set to default answer
    best_guest = np.where(predictions == np.amax(predictions))[0]
    if predictions[best_guest] > confidence:
        choosen_class = best_guest[0]

    return intents[choosen_class]
  
def ask_bot(query):
    intent = classify_query(query)
    logic = 'intent == "' + intent + '"'
    return responses.query(logic).sample(n = 1)

while True:
    try:
        bot_output = ask_bot(input())
        print(bot_output)

    except(KeyboardInterrupt, EOFError, SystemExit):
        break
