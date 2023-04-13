import numpy as np
from typing import List
import sys
sys.path.append('hw1/')
from model import Model
import re
import torch
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import random
from loss import NEGLoss

import config

class CBOW(torch.nn.Module):
    '''
    CBOW model
    '''
    def __init__(self, vocab_size, embedding_dim, id2word, word_counts, NEG_SAMPLING=False):
        '''
        Init the CBOW model

        @param vocab_size: size of the vocabulary
        @param embedding_size: size of the embedding
        '''
        super(CBOW, self).__init__()
        self.lin_layer = torch.nn.Linear(vocab_size, embedding_dim)
        self.embedding = torch.nn.Linear(embedding_dim, vocab_size)
        
        if NEG_SAMPLING:
            self.loss_function = NEGLoss(id2word, word_counts)
        else:
            self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, input):
        '''
        Encode the input with the dense layer and decode it with the linear layer

        @param input: input tensor of shape (batch_size, seq_len)
        '''
        hidden = self.lin_layer(input)
        output_embeddings = self.embedding(hidden)
        if isinstance(self.loss_function, NEGLoss):
            output = torch.nn.functional.log_softmax(output_embeddings, dim=-1)
        else:
            # CrossEntropyLoss applies log_softmax internally
            output = output_embeddings
        return output

def train_cbow(dataset, vocab, embedding_size=300, batch_size=32, num_epochs=30, learning_rate=1e-4, lr_decay=0.9):
    '''
    Train the cbow model

    @param data: list of sentences
    @param vocab: vocabulary
    @param embedding_size: size of the embedding
    @param batch_size: size of the batch
    @param num_epochs: number of epochs
    @param learning_rate: learning rate
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = CBOW(len(vocab), embedding_size).to(device)
    if config.RESUME_CBOW:
        model.load_state_dict(torch.load('./model/cbow.pth'))
        print('Model loaded')
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float('inf')
    X, y = dataset.generate_data(batch_size)
    
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        epoch_loss = 0

        i = 1
        bar = tqdm(zip(X, y), total=len(X))
        
        for X_, y_ in bar:  #dataset.iterate_cbow(batch_size):
            X_ = X_.to(device)
            y_ = y_.to(device)
            optimizer.zero_grad()
            outputs = model(X_)
            loss = loss_fn(outputs, y_)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            i += 1
        learning_rate *= lr_decay
        for g in optimizer.param_groups:
            g['lr'] = learning_rate
        
        print("lr:", learning_rate)

        
        print(f'Loss: {epoch_loss}')
        print()

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), './model/cbow.pth')
            print('Model saved')

    return model
