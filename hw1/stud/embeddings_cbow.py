import numpy as np
import torch
from loss_negloss import NEGLoss
import config
from embeddings_interface import Embedding

class CBOW(torch.nn.Module, Embedding):
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
        self.embeddings = torch.nn.Linear(embedding_dim, vocab_size)
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
        output_embeddings = self.embeddings(hidden)
        if isinstance(self.loss_function, NEGLoss):
            output = torch.nn.functional.log_softmax(output_embeddings, dim=-1)
        else:
            # CrossEntropyLoss applies log_softmax internally
            output = output_embeddings
        return output
    
    def get_embeddings(self):
        return self.embeddings.weight.detach().cpu()

