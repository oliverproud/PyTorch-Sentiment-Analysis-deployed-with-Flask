# Contains the functions to load the state dict into the model and data preprocessing
import torch
import torch.nn as nn
import spacy
import os
import pickle
import torchtext
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import requests
import re

nlp = spacy.load("en_core_web_sm")

class CNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim,
                 n_filters, filter_sizes, output_dim,
                 dropout, pad_idx):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)

        '''
        ModuleList means an arbirtary sized list of filter sizes can be provided
        and the list comprehension will create conv layers for each of the filters
        '''
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, out_channels = n_filters,
                                              kernel_size = (fs, embedding_dim)) for fs in filter_sizes])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):

        '''
        In PyTorch RNNs want the input with batch dim second, CNNs want the batch dim first
        we permute the input to make it the right shape for the CNN
        '''
        text = text.permute(1, 0)

        # Text passed through embedding layer to get embeddings
        embedded = self.embedding(text)

        '''
        A conv layer wants the second dim of the input to be a channel dim
        text does not have a channel dim, so the tensor is unsqueezed to create one
        '''
        embedded = embedded.unsqueeze(1)

        # Iterates through the list of conv layers applying each conv layer to get list of conv outputs
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        '''
        Conv outputs are passed through a max pooling that takes the maximum value over a dimension
        the idea being that the "maximum value" is the most important feature for determining the sentiment
        which corresponds to the most important n-gram in the review
        '''
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        '''
        The model has 100 filters of 3 different sizes, therefore 300 n-grams that could be important
        which we concatenate into a single vector and pass through a dropout layer and finally a linear layer
        (NOTE: dropout is set to 0 during inference time)
        '''
        cat = self.dropout(torch.cat(pooled, dim = 1))

        # passed through linear layer to make predictions
        return self.fc(cat)

data_dir = 'static/models'
s3_model_url = 'https://sent-model.s3.eu-west-2.amazonaws.com/conv-sentiment_model1.pt'

path_to_model = os.path.join(data_dir, 'conv-sentiment_model1.pt')
if not os.path.exists(path_to_model):
    print("Model weights not found, downloading from S3...")
    os.makedirs(os.path.join(data_dir), exist_ok=True)
    filename = Path(path_to_model)
    r = requests.get(s3_model_url)
    filename.write_bytes(r.content)

model = CNN(25002, 300, 100, [3,4,5], 1, 0.55, 1)
model.load_state_dict(torch.load(path_to_model, map_location='cpu'))

s3_word_dict_url = 'https://sent-model.s3.eu-west-2.amazonaws.com/word_dict.pkl'

path_to_dict = os.path.join(data_dir, 'word_dict.pkl')
if not os.path.exists(path_to_dict):
    print("Word dict not found, downloading from S3...")
    os.makedirs(os.path.join(data_dir), exist_ok=True)
    filename = Path(path_to_dict)
    r = requests.get(s3_word_dict_url)
    filename.write_bytes(r.content)

with open(path_to_dict, 'rb') as f:
    TEXT = pickle.load(f)

def predict_sentiment(sentence, model=model,min_len = 5):

    model.eval()

    sentence = sentence.lower()
    # Remove punctuation
    sentence = re.sub(r'[^\w\s]','', sentence)

    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]

    '''
    If the length of the sentence is shorter than the length of the largest filter
    then the sentence must be padded to the length of the largest filter
    '''
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [TEXT.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed)
    tensor = tensor.unsqueeze(1)
    prediction = torch.sigmoid(model(tensor))

    probs = [{'name': index, 'prob': prediction.item()} for index in np.argsort(prediction.item())]

    return (sentence, probs)
