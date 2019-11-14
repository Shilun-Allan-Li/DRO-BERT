# -*- coding: utf-8 -*-
"""
@author: Allan Li
"""
import argparse
import sys
import torch
import random
import numpy as np
import torch.nn as nn
from pytorch_pretrained_bert import BertModel

class Net(nn.Module):
    '''                                                                                                
    a (maybe bi-directional) LSTM layer on top of BERT to preform sentiment analysis
    '''
    def __init__(self, dropout, hidden_dim, model, device, case = 'cased'):
        super().__init__()
        assert case in {'cased', 'uncased'}
        assert model in {'base', 'large'}
        embedding_dim = 769 if model == 'base' else 1024
        self.bert = BertModel.from_pretrained('bert-{}-{}'.format(model, case))
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           bidirectional=bidirectional,
                           dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        encoded_layers, _ = self.bert(x)
        enc = encoded_layers[-1]
        #ToDo: enc.size()[1] =? length of the sentences
        packed_embedded = nn.utils.rnn.pack_padded_sequence(enc, enc.size()[1], batch_first=True)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors
        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout
        # hidden = [batch size, hid dim * num directions]
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        y_hat = self.fc(hidden)
        return y_hat
