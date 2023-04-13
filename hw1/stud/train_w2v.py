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
from skipgram import SkipGram
from cbow import CBOW



class Trainer():
    def __init__(self, model, optimizer):

        self.device = config.device

        self.model = model
        self.optimizer = optimizer

        # starts requires_grad for all layers
        self.model.train()  # we are using this model for training (some layers have different behaviours in train and eval mode)
        self.model.to(self.device)  # move model to GPU if available

    def train(self, train_dataset, output_folder, epochs=1):

        train_loss = 0.0
        for epoch in range(epochs):
            epoch_loss = 0.0
            len_train = 0

            # each element (sample) in train_dataset is a batch
            for step, sample in tqdm(enumerate(train_dataset), desc="Batch", leave=False):
                # inputs in the batch
                inputs = sample['inputs']
                # outputs in the batch
                targets = sample['targets'].to(self.device)

                # one_hot_input : batch size X vocab_size
                one_hot_input = torch.zeros((inputs.shape[0], config.vocab_size), device=self.device)
                # sets the ones corresponding to the input word
                for i, x in enumerate(inputs):
                    one_hot_input[i, x] = 1

                output_distribution = self.model(one_hot_input)
                loss = self.model.loss_function(output_distribution, targets)  # compute loss
                # calculates the gradient and accumulates
                loss.backward()  # we backpropagate the loss
                # updates the parameters
                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()
                len_train += 1
            avg_epoch_loss = epoch_loss / len_train

            print('Epoch: {} avg loss = {:0.10f}'.format(epoch, avg_epoch_loss))

            train_loss += avg_epoch_loss
            torch.save(self.model.state_dict(),
                       os.path.join(output_folder, 'state_{}.pt'.format(epoch)))  # save the model state

        avg_epoch_loss = train_loss / epochs
        return avg_epoch_loss
    
    
    

torch.manual_seed(42)

print('Using Negative Sampling: ', config.NEGATIVE_SAMPLING)

dataset = Word2VecDataset(config.train_file, config.vocab_size, config.UNK_TOKEN, window_size=5)
model = None
if config.ALGORITHM == config.USE_SKIPGRAM:
    model = SkipGram(config.vocab_size, embedding_dim=300, id2word=dataset.id2word,
                     word_counts=dataset.frequency, NEG_SAMPLING=config.NEGATIVE_SAMPLING)
elif config.ALGORITHM == config.USE_CBOW:
    model = CBOW(config.vocab_size, embedding_dim=300, id2word=dataset.id2word,
                    word_counts=dataset.frequency, NEG_SAMPLING=config.NEGATIVE_SAMPLING)
else:
    raise ValueError('Invalid algorithm')

# define an optimizer (stochastic gradient descent) to update the parameters
optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.lr_decay)

if config.TRAIN:
    trainer = Trainer(model, optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size) #it batches data for us
    avg_loss = trainer.train(dataloader, config.OUTPUT_PATH, epochs=config.num_epochs)

if config.EVALUATE:
    for epoch in [0, config.num_epochs // 2, config.num_epochs - 1]:
        ## load model from checkpoint
        model.load_state_dict(torch.load(os.path.join(config.OUTPUT_PATH, 'state_{}.pt'.format(epoch))))

        # set the model in evaluation mode
        # (disables dropout, does not update parameters and gradient)
        model.eval()

        # retrieve the trained embeddings
        embeddings = model.get_embeddings()
        
        # pick some words to visualise
        words = ['dog', 'horse', 'animals', 'france', 'italy', 'parents']

        # perform PCA to reduce our 300d embeddings to 2d points that can be plotted
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(embeddings.detach().t().cpu()) # .t() transpose the embeddings

        indexes = [dataset.word2id[x] for x in words]
        points = [pca_result[i] for i in indexes]
        for i,(x,y) in enumerate(points):
            plt.plot(x, y, 'ro')
            plt.text(x, y, words[i], fontsize=12) # add a point label, shifted wrt to the point
        plt.title('epoch {}'.format(epoch))
        plt.show()