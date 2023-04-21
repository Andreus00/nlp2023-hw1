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
# import gensim
# from gensim import downloader as api


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    return StudentModel()


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

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.lin_layer = torch.nn.Linear(in_channels, out_channels)
        self.activation = torch.nn.ReLU()
        self.shortcut = torch.nn.Linear(in_channels, out_channels) if in_channels != out_channels else torch.nn.Identity()
        self.bn = torch.nn.BatchNorm1d(config.batch_size)

    def forward(self, x):
        residual = x
        x = self.lin_layer(x)
        x += self.shortcut(residual)
        x = x.permute(1, 0, 2)
        # if x.shape[1] != config.batch_size:
        #     last = x.shape[1]
        #     x = torch.nn.functional.pad(x, (0, 0, 0, config.batch_size - last))
        #     x = self.bn(x)
        #     # remove padding
        #     x = x[:last]

        x = self.bn(x)

        x = x.permute(1, 0, 2)
        x = self.activation(x)
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

class StudentModel(Model, torch.nn.Module):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary

    def __init__(self):
        # first load the embeddings
        super().__init__()
        embeddings_path = None
        vocab_path = None
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
            self.conv1 = torch.nn.Conv1d(in_channels=config.embedding_size, out_channels=512, kernel_size=7, stride=1, padding=3)
            self.bn_1 = torch.nn.BatchNorm1d(512)
            self.non_lin_1 = torch.nn.ReLU()
            self.lstm_1 = torch.nn.LSTM(input_size=512, hidden_size=2048, num_layers=2, batch_first=True, bidirectional=True)
            self.flatten_1 = torch.nn.Flatten()
            self.lin_layer_1 = torch.nn.Linear(1024, 512)
            self.non_lin_1_2 = torch.nn.ReLU()


            self.conv2 = torch.nn.Conv1d(in_channels=config.embedding_size, out_channels=512, kernel_size=5, stride=1, padding=2)
            self.bn_2 = torch.nn.BatchNorm1d(512)
            self.non_lin_2 = torch.nn.ReLU()
            self.lstm_2 = torch.nn.LSTM(input_size=512, hidden_size=2048, num_layers=2, batch_first=True, bidirectional=True)
            self.flatten_2 = torch.nn.Flatten()
            self.lin_layer_2 = torch.nn.Linear(1024, 512)
            self.non_lin_2_2 = torch.nn.ReLU()

            self.conv3 = torch.nn.Conv1d(in_channels=config.embedding_size, out_channels=512, kernel_size=3, stride=1, padding=1)
            self.bn_3 = torch.nn.BatchNorm1d(512)
            self.non_lin_3 = torch.nn.ReLU()
            self.lstm_3 = torch.nn.LSTM(input_size=512, hidden_size=2048, num_layers=2, batch_first=True, bidirectional=True)
            self.flatten_3 = torch.nn.Flatten()
            self.lin_layer_3 = torch.nn.Linear(1024, 512)
            self.non_lin_3_2 = torch.nn.ReLU()

            self.final_lin_layer = torch.nn.Linear(512*3, 11)

        elif config.MODEL == 3:
            # multiply the inputs my a matrix of weights
            layers += [
                torch.nn.LSTM(input_size=config.embedding_size, hidden_size=4096, num_layers=2, batch_first=True, bidirectional=True),
                torch.nn.ReLU(),
                torch.nn.Linear(4096*2, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 11)
            ]
        
        self.num_layers = len(layers)
        for i in range(self.num_layers):
            name = f"layer_{i}"
            setattr(self, name, layers[i])

        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-100)

        if config.RESUME != None:
            self.load_state_dict(torch.load(config.RESUME, map_location=torch.device('cpu')))
            self.eval()


    def forward(self, input):
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
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!
        data = []
        for i in range(len(tokens)):
            cur_sentence = tokens[i].copy()
            for k in range(len(cur_sentence)):
                cur_sentence[k] = self.get_key(cur_sentence[k])
            data.append(cur_sentence)

        if len(data) < config.batch_size:
            data += [[self.vocab[config.UNK_TOKEN]]] * (config.batch_size - len(data))
        length = [len(x) for x in data]
        inputs = torch.tensor(data)
        
        # pad inputs
        padded_seq = pad_sequence(inputs, batch_first=True, padding_value=config.vocab_size - 1)
        
        output = self.forward(inputs)

        output = torch.argmax(output, dim=2).tolist()

        return output

    def loss_function(self, output, target):
        # STUDENT: implement here your loss function
        return self.loss(output, target)

    def get_key(self, key):
        if key in self.vocab:
            return self.vocab[key]
        else:
            return self.vocab[config.UNK_TOKEN]
    
    def unfreeze_embeddings(self, lr=1e-4):
        for param in self.embedding.parameters():
            param.requires_grad = True



if __name__ == "__main__":
    model = StudentModel()