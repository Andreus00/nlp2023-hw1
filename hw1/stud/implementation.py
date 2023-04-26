import numpy as np
from typing import List
import sys
sys.path.append('hw1/')
sys.path.append('hw1/stud/')
from model import Model
import re
import torch
from tqdm import tqdm
import json
import random
from embeddings_cbow import CBOW
from embeddings_skipgram import SkipGram
from dataset_word2vec import Word2VecDataset
import config
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import classes
from fasttext import FastTextWrapper

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    return StudentModel().to(device)


class RandomBaseline(Model):
    options = [
        (22458, "B-ACTION"),
        (13256, "B-CHANGE"),
        (2711, "B-POSSESSION"),
        (6405, "B-SCENARIO"),
        (3024, "B-SENTIMENT"),
        (457, "I-ACTION"),
        (583, "I-CHANGE"),
        (30, "I-POSSESSION"),
        (505, "I-SCENARIO"),
        (24, "I-SENTIMENT"),
        (463402, "O")
    ]

    def __init__(self):
        self._options = [option[1] for option in self.options]
        self._weights = np.array([option[0] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return [
            [str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x]
            for x in tokens
        ]

class ExtractOutputFromLSTM(torch.nn.Module):
    '''
    Class used after an LSTM layer to extract the output from the LSTM.
    '''
    def __init__(self, *args, **kwargs) -> None:
        super(ExtractOutputFromLSTM, self).__init__(*args, **kwargs)

    def forward(self, x):
        return x[0]

class ResidualBlock(torch.nn.Module):
    '''
    Implementation of a residual block.
    '''
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.lin_layer = torch.nn.Linear(in_channels, out_channels)
        self.shortcut = torch.nn.Linear(in_channels, out_channels) if in_channels != out_channels else torch.nn.Identity()
        self.activation = torch.nn.ReLU()

        # normalize the input
        if config.NORM_IN_RESBLOCK:
            self.norm = torch.nn.LayerNorm(out_channels)


    def forward(self, x):
        residual = x
        x = self.lin_layer(x)
        x += self.shortcut(residual)
        x = self.activation(x)
        if config.NORM_IN_RESBLOCK:
            x = self.norm(x)
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

class StudentModel(Model, torch.nn.Module):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary

    def __init__(self):
        '''
        Inside this init functions, embeddings are loaded and the classifier is created based
        on the config file.
        '''
        super().__init__()
        embeddings_path = None
        vocab_path = None

        # first load the embeddings
        if not config.MODEL_HANDLES_OOV:    
            if config.ALGORITHM == config.USE_SKIPGRAM:
                embeddings_path = "model/skipgram/embeddings.pt"
                vocab_path = "model/skipgram/vocab.json"
            elif config.ALGORITHM == config.USE_CBOW:
                embeddings_path = "model/cbow/embeddings.pt"
                vocab_path = "model/cbow/vocab.json"
            elif config.ALGORITHM == config.USE_GLOVE:
                embeddings_path = "model/glove/glove-wiki-gigaword-300.pt"
                vocab_path = "model/glove/glove-wiki-gigaword-300-vocab.json"
            elif config.ALGORITHM == config.USE_W2V_GENSIM:
                embeddings_path = "model/w2v/word2vec-google-news-300.pt"
                vocab_path = "model/w2v/word2vec-google-news-300.json"
            self.embedding: torch.nn.Embedding = torch.load(embeddings_path)
            for param in self.embedding.parameters():
                param.requires_grad = False
            with open(vocab_path) as f:
                self.vocab = json.load(f)
        else:
            if config.ALGORITHM == config.USE_FASTTEXT:
                vectors_path = 'model/fasttext/fasttext-wiki-news-subwords-300-vectors.pt'
                vocab_path = 'model/fasttext/fasttext-wiki-news-subwords-300-vocab.json'
                vectors_ngrams_path = 'model/fasttext/fasttext-wiki-news-subwords-300-vectors_ngrams.pt'
                self.embedding = FastTextWrapper(vectors_path, vocab_path, vectors_ngrams_path, 3, 6, 2000000)
        

        if config.USE_POS_TAGGING:
            self.pos_embedding = torch.nn.Embedding(len(classes.pos2int), 300, padding_idx=classes.pos2int["PAD"])
                

        # then create the lstm classifier
        layers = []        
        if config.MODEL == 0:

            layers += [
                torch.nn.LSTM(input_size=config.embedding_size, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True),
                torch.nn.GELU(),
                torch.nn.Linear(1024, 11)
            ]
        elif config.MODEL == 1:
            # multiply the inputs my a matrix of weights
            layers += [
                torch.nn.LSTM(input_size=config.embedding_size, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True),
                torch.nn.LSTM(input_size=1024, hidden_size=2048, num_layers=1, batch_first=True, bidirectional=True),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.5),
                ResidualBlock(4096, 4096),
                torch.nn.Dropout(p=0.5),
                ResidualBlock(4096, 1024),
                torch.nn.Dropout(p=0.5),
                torch.nn.Linear(1024, 11)
            ]
        elif config.MODEL == 2:
            # multiply the inputs my a matrix of weights
            layers += [
                torch.nn.LSTM(input_size=config.embedding_size, hidden_size=4096, num_layers=2, batch_first=True, bidirectional=True),
                torch.nn.ReLU(),
                torch.nn.Linear(4096*2, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 11)
            ]
        elif config.MODEL == 3:
            # multiply the inputs my a matrix of weights
            layers += [
                torch.nn.LSTM(input_size=config.embedding_size, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True),
                torch.nn.LSTM(input_size=1024, hidden_size=2048, num_layers=1, batch_first=True, bidirectional=True),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.5),
                ResidualBlock(4096, 4096),
                torch.nn.Dropout(p=0.5),
                ResidualBlock(4096, 4096),
                torch.nn.Dropout(p=0.5),
                ResidualBlock(4096, 4096),
                torch.nn.Dropout(p=0.5),
                ResidualBlock(4096, 1024),
                torch.nn.Dropout(p=0.5),
                ResidualBlock(1024, 1024),
                torch.nn.Dropout(p=0.5),
                ResidualBlock(1024, 1024),
                torch.nn.Dropout(p=0.5),
                ResidualBlock(1024, 1024),
                torch.nn.Dropout(p=0.5),
                torch.nn.Linear(1024, 11)
            ]
        elif config.MODEL == 4:
            ## Use this only with fasttext and bigrams
            # layers += [
            #     torch.nn.LSTM(input_size=config.embedding_size, hidden_size=2048, num_layers=2, batch_first=True, bidirectional=True),
            #     ExtractOutputFromLSTM(),
            #     ResidualBlock(4096, 4096),
            #     torch.nn.Dropout(p=0.5),
            # ]
            layers += [
                torch.nn.LSTM(input_size=config.embedding_size, hidden_size=1024, num_layers=2, batch_first=True, bidirectional=True),
                ExtractOutputFromLSTM(),
                ResidualBlock(1024 * 2, 1024),
                torch.nn.Dropout(p=0.5),
            ]

            self.main_layers = torch.nn.Sequential(*layers)

            self.bigrams_layers = torch.nn.Sequential(
                torch.nn.LSTM(input_size=config.embedding_size, hidden_size=1024, num_layers=2, batch_first=True, bidirectional=True),
                ExtractOutputFromLSTM(),
                ResidualBlock(1024 * 2, 1024),
                torch.nn.Dropout(p=0.5),
            )
            # self.bigrams_layers = torch.nn.Sequential(
            #     torch.nn.LSTM(input_size=config.embedding_size, hidden_size=2048, num_layers=2, batch_first=True, bidirectional=True),
            #     ExtractOutputFromLSTM(),
            #     ResidualBlock(4096, 4096),
            #     torch.nn.Dropout(p=0.5),
            # )

            self.union_layers = torch.nn.Sequential(
                torch.nn.Linear(1024*2, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.5),
                torch.nn.Linear(512, 11)
            )


        if not config.MODEL == 4:
            self.num_layers = len(layers)
            for i in range(self.num_layers):
                name = f"layer_{i}"
                setattr(self, name, layers[i])

        
        # then create the loss function
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-100)


        # finally, if required, load the model's checkpoints
        if config.RESUME != None:
            self.load_state_dict(torch.load(config.RESUME, map_location=torch.device(device)))
            self.eval()


    def forward(self, input):
        '''
        Forward based on the selected model
        '''
        if config.MODEL == 4:
            o1 = self.embedding(input[0])
            o2 = self.embedding(input[1])
            if config.USE_POS_TAGGING:
                pos = input[2]
                pos = self.pos_embedding(pos)
                o1 += pos
            o1 = self.main_layers(o1)
            o2 = self.bigrams_layers(o2)
            output = torch.cat((o1, o2), dim=2)
            output = self.union_layers(output)
        else:
            output = self.embedding(input)      
            for i in range(self.num_layers):
                name = f"layer_{i}"
                layer = getattr(self, name)
                if isinstance(layer, torch.nn.LSTM):
                    output, (h, c) = layer(output)
                else:
                    output = layer(output)
        return output

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        '''
        Predict the labels for the given tokens
        '''
        data = []
        for i in range(len(tokens)):
            cur_sentence = tokens[i].copy()
            for k in range(len(cur_sentence)):
                cur_sentence[k] = self.get_key(cur_sentence[k])
            data.append(cur_sentence)

        length = [len(x) for x in data]
        inputs = [torch.tensor(x, device=config.device) for x in data]
        
        # pad inputs
        padded_seq = pad_sequence(inputs, batch_first=True, padding_value=config.vocab_size - 1)
        
        output = self.forward(padded_seq)

        output = torch.argmax(output, dim=2).tolist()

        for i in range(len(tokens)):
            output[i] = output[i][:len(tokens[i])]

        for i in range(len(output)):
            for j in range(len(output[i])):
                output[i][j] =  classes.int2class[output[i][j]]

        return output
    

    def loss_function(self, output, target):
        '''
        Returns the loss for the given output and target
        '''
        return self.loss(output, target)

    def get_key(self, key):
        '''
        Returns the index of the key in the vocabulary.
        '''
        if config.MODEL_HANDLES_OOV:
            return self.embedding.get_index(key)
        else:
            if key in self.vocab:
                return self.vocab[key]
            else:
                return self.vocab[config.UNK_TOKEN]
    
    def unfreeze_embeddings(self, lr=1e-4):
        '''
        Unfreeze the embeddings layer and set the learning rate to config.EMB_LR
        '''
        if config.MODEL_HANDLES_OOV:
            self.embedding.unfreeze_embeddings()
        else:
            for param in self.embedding.parameters():
                param.requires_grad = True
        self.optimizer = torch.optim.Adam(self.embedding.parameters(), lr=config.EMB_LR)



if __name__ == "__main__":
    model = StudentModel()
    model.eval()
    print(model.predict([["hello", "dark_souls", "dark_souls"]]))