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
from dataset_word2vec import Word2VecDataset
from loss_negloss import NEGLoss
from tqdm.auto import tqdm
from embeddings_skipgram import SkipGram
from embeddings_cbow import CBOW
from eval_utils import plot_embeddings_close_to_word
from trainer import Trainer

torch.manual_seed(42)

## load the dataset and the model
dataset = Word2VecDataset(config.train_file, config.vocab_size, config.UNK_TOKEN, window_size=5)
model = None
if config.ALGORITHM == config.USE_SKIPGRAM:
    model = SkipGram(config.vocab_size, embedding_dim=config.embedding_size, id2word=dataset.id2word,
                     word_counts=dataset.frequency, NEG_SAMPLING=config.NEGATIVE_SAMPLING)
elif config.ALGORITHM == config.USE_CBOW:
    model = CBOW(config.vocab_size, embedding_dim=config.embedding_size, id2word=dataset.id2word,
                    word_counts=dataset.frequency, NEG_SAMPLING=config.NEGATIVE_SAMPLING)
else:
    raise ValueError('Invalid algorithm')

# define an optimizer to update the parameters
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)


## Train the model
if config.TRAIN:
    trainer = Trainer(model, optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size) #it batches data for us
    avg_loss = trainer.train(dataloader, config.OUTPUT_W2V_PATH, epochs=config.num_epochs)



## Plot the embeddings

if config.EVALUATE:
    
    for epoch in [0, config.num_epochs // 2, config.num_epochs - 1]:
        ## load model from checkpoint
        model.load_state_dict(torch.load(os.path.join(config.OUTPUT_W2V_PATH, 'state_{}.pt'.format(epoch))))

        # set the model in evaluation mode
        # (disables dropout, does not update parameters and gradient)
        model.eval()

        # retrieve the trained embeddings
        embeddings = model.get_embeddings()
        print('embeddings shape: ', embeddings.shape)
        
        # pick some words to visualise
        words = ['dog', 'horse', 'animals', 'france', 'italy', 'parents']

        # perform PCA to reduce our 300d embeddings to 2d points that can be plotted
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(embeddings) # .t() transpose the embeddings

        indexes = [dataset.word2id[x] for x in words]
        points = [pca_result[i] for i in indexes]
        for i,(x,y) in enumerate(points):
            plt.plot(x, y, 'ro')
            plt.text(x, y, words[i], fontsize=12) # add a point label, shifted wrt to the point
        plt.title('epoch {}'.format(epoch))
        plt.show()

        # plot the closest words to the selected words
        plot_embeddings_close_to_word(dataset.word2id, embeddings, embeddings[dataset.word2id['february']])

