import pandas
import numpy as np
import pickle
from fastapi import FastAPI
from mangum import Mangum
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Embedding
import keras 
from tf2crf import CRF, ModelWithCRFLoss
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from flask import Flask, request, jsonify
import re

app = FastAPI(
    title="My App",
    description="My description",
    version="1.0",
    docs_url='/docs',
    openapi_url='/openapi.json',
    redoc_url=None
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://*","http://*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
handler = Mangum(app)

def open_object(filename):
    with open(filename, 'rb') as inp:
        opened = pickle.load(inp)
        return opened  

class CustomLayer(keras.layers.Layer):
    def __init__(self, sublayer, **kwargs):
        super().__init__(**kwargs)
        self.sublayer = sublayer

    def call(self, x):
        return self.sublayer(x)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "sublayer": keras.saving.serialize_keras_object(self.sublayer),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        sublayer_config = config.pop("sublayer")
        sublayer = keras.saving.deserialize_keras_object(sublayer_config)
        return cls(sublayer, **config)

words = open_object(r'words.pkl')
word2idx = open_object(r'word2idx.pkl')
tag2idx = open_object(r'tag2idx.pkl')
idx2tag = open_object(r'idx2tag.pkl')
n_words = len(words)
n_tags = len(tag2idx)
input = Input(shape=(50,))
word_embedding_size = 100
output = Embedding(input_dim=n_words, input_length=50, output_dim = word_embedding_size, trainable=True, mask_zero=True)(input)
model = Bidirectional(LSTM(units=word_embedding_size, 
                           return_sequences=True, 
                           dropout=0.5, 
                           recurrent_dropout=0.5,
                           kernel_initializer=keras.initializers.he_normal()))(output)
model = LSTM(units=word_embedding_size * 2, 
             return_sequences=True, 
             dropout=0.5, 
             recurrent_dropout=0.5, 
             kernel_initializer=keras.initializers.he_normal())(model)
output = Dense(n_tags, activation=None)(model)
crf = CustomLayer(CRF())
output = crf(output)
base_model = Model(input, output)
model = ModelWithCRFLoss(base_model)
model.load_weights('simple_model/variables/variables')
model.compile(optimizer=keras.optimizers.Adam())

class Paragraph(BaseModel):
    par: str

def SeparateParagraph(parag):
    sens = parag.split('.')
    result = []
    for sen in sens[:-1]:
        result_sen = []
        print(sen)
        if sen[0]==' ':
            sen = sen[1:]
        for word in sen.split(' '):
            if word[0]=='(':
                result_sen.append("(")
                word=word[1:] 
            if word[-1]==',':
                result_sen.append(word[:-1])
                result_sen.append(',')
            elif word[-2:]=="’s" or word[-2:]=="'s":
                result_sen.append(word[:-2])
                result_sen.append("'s")
            elif word[-1]==')':
                result_sen.append(word[:-1])
                result_sen.append(')')
            else:
                result_sen.append(word)
        result.append(result_sen)
    return result


@app.post("/result/")
def Inference(p:Paragraph):
    p1 = SeparateParagraph(p.par)
    ex = [[word2idx[w] if w in word2idx else n_words-1 for w in s] for s in p1]
    ex = pad_sequences(maxlen=50, sequences=ex, padding="post",value=n_words-1)
    pred = model.predict(ex).tolist()
    dict_list = []
    for test_id in range(len(p1)):
        obj = []
        for idx, (a, w,p) in enumerate(zip(p1[test_id], ex[test_id],pred[test_id])):
            obj.append([a, idx2tag[p]])
        dict_list.append(obj)
    return dict_list