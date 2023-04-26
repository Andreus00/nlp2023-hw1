import torch.nn as nn
import torch
import config
from dataset_word2vec import Word2VecDataset
from loss_negloss import NEGLoss
from embeddings_interface import Embedding


class SkipGram(nn.Module, Embedding):
    '''
    SkipGram model.

    Partially taken from the notebook.
    '''

    def __init__(self, vocabulary_size, embedding_dim, id2word, word_counts, NEG_SAMPLING=False):
        super(SkipGram, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Linear(self.vocabulary_size, self.embedding_dim)
        self.output_weights = nn.Linear(self.embedding_dim, self.vocabulary_size)
        if NEG_SAMPLING:
            self.loss_function = NEGLoss(id2word, word_counts)
        else:
            self.loss_function = nn.CrossEntropyLoss()

    def forward(self, input_idx):
        input_embeddings = self.embeddings(input_idx)  # compute the embeddings for the input words
        output_embeddings = self.output_weights(input_embeddings)
        if isinstance(self.loss_function, NEGLoss):
            output = nn.functional.log_softmax(output_embeddings, dim=-1)
        else:
            # CrossEntropyLoss applies log_softmax internally
            output = output_embeddings
        return output
    

    def get_embeddings(self):
        return self.embeddings.weight.T.detach().cpu()
    

    def embed_words(self, words):
        # This method embeds a list of words
        # @param words: list of words to embed
        # @return: tensor of shape (len(words), embedding_dim)
        embeddings = self.embeddings(words)
        return embeddings
    
