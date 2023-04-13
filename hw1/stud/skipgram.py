from sklearn.decomposition import PCA
from torch import FloatTensor as FT
from torch import LongTensor as LT
import matplotlib.pyplot as plt
import torch.nn as nn
import collections
import numpy as np
import torch
import json
import os
import re
import config
from word2vec_dataset import Word2VecDataset
from loss import NEGLoss
from tqdm.auto import tqdm



class SkipGram(nn.Module):

    def __init__(self, vocabulary_size, embedding_dim, id2word, word_counts, NEG_SAMPLING=False):
        super(SkipGram, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim

        # matrix W
        self.embeddings = nn.Linear(self.vocabulary_size, self.embedding_dim)
        # matrix W'
        self.output_weights = nn.Linear(self.embedding_dim, self.vocabulary_size)

        if NEG_SAMPLING:
            self.loss_function = NEGLoss(id2word, word_counts)
        else:
            self.loss_function = nn.CrossEntropyLoss()

    def forward(self, input_idx):
        # This method defines the outputs of a forward pass on the model
        input_embeddings = self.embeddings(input_idx)  # compute the embeddings for the input words
        output_embeddings = self.output_weights(input_embeddings)
        if isinstance(self.loss_function, NEGLoss):
            output = nn.functional.log_softmax(output_embeddings, dim=-1)
        else:
            # CrossEntropyLoss applies log_softmax internally
            output = output_embeddings
        return output
    


