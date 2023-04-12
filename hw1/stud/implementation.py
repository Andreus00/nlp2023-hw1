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


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

batch_size = 256
learning_rate = 3e-3
lr_decay=1
num_epochs = 100
embedding_size = 300
window_size = 5


TESTING = True

GENERATE_VOCAB = False
RESUME_WORD2VEC = True if TESTING else False
TRAIN_WORD2VEC = False if TESTING else True


def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    return RandomBaseline()


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


class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary

    def __init__(self):
        # first load word2vec
        
        pass

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!
        pass


class Word2Vec(torch.nn.Module):
    '''
    Word2Vec model
    '''
    def __init__(self, vocab_size, embedding_size):
        '''
        Init the Word2Vec model

        @param vocab_size: size of the vocabulary
        @param embedding_size: size of the embedding
        '''
        super(Word2Vec, self).__init__()
        self.embeding = torch.nn.Embedding(vocab_size, embedding_size)
        self.lin_layer = torch.nn.Linear(embedding_size, 128)
        self.relu = torch.nn.ReLU()
        self.lin_layer2 = torch.nn.Linear(128, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input):
        '''
        Encode the input with the dense layer and decode it with the linear layer

        @param input: input tensor of shape (batch_size, seq_len)
        '''
        hidden = self.embeding(input)
        logits = self.lin_layer(hidden)
        logits = self.relu(logits)
        logits = self.lin_layer2(logits)
        out = self.softmax(logits)
        return out

def train_word2vec(dataset, vocab, embedding_size=300, batch_size=32, num_epochs=30, learning_rate=1e-4, lr_decay=0.999):
    '''
    Train the Word2Vec model

    @param data: list of sentences
    @param vocab: vocabulary
    @param embedding_size: size of the embedding
    @param batch_size: size of the batch
    @param num_epochs: number of epochs
    @param learning_rate: learning rate
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = Word2Vec(len(vocab), embedding_size).to(device)
    if RESUME_WORD2VEC:
        model.load_state_dict(torch.load('./model/word2vec.pth'))
        print('Model loaded')
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        epoch_loss = 0

        i = 1
        for X_, y_ in dataset.iterate_w2v(batch_size):
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
            torch.save(model.state_dict(), './model/word2vec.pth')
            print('Model saved')

    return model


def one_hot(word_idx, length):
    '''
    One hot encoding of a word

    @param word: word to encode
    @param vocab: vocabulary
    '''
    one_hot = torch.zeros(length)
    one_hot[word_idx] = 1
    return one_hot

def get_sentences_labels(path):
    '''
    Get the sentences and labels from the dataset

    @param path: path to the dataset
    '''
    sentences = []
    labels = []

    with open(path, 'r') as json_file:
        json_list = list(json_file)

    for json_str in tqdm(json_list):
        result = json.loads(json_str)
        sentences.append(result['tokens'])
        labels.append(result['labels'])

    return sentences, labels


def generate_vocab_sentences_labels(path, save=True):
    gist_file = open("./data/gist_stopwords.txt", "r")
    stopwords = set()
    try:
        content = gist_file.read()
        stopwords = set(content.split(","))
    finally:
        gist_file.close()

    print("Stopwords: ", len(stopwords))

    sentences = []
    labels = []
    vocab = []
    json_list = []
    word2freq = {}

    with open(path, 'r') as json_file:
        json_list = list(json_file)

    for json_str in tqdm(json_list):
        result = json.loads(json_str)
        sentences.append(result['tokens'])
        labels.append(result['labels'])
        for token in result['tokens']:
            token = token.lower()
            if (any([not c.isalnum() for c in token]) and not re.match(r"(?=\S*['-])([a-zA-Z'-]+)", token)) or token in stopwords:
                continue
            vocab.append(token)
            if token in word2freq:
                word2freq[token] += 1
            else:
                word2freq[token] = 1

    if save==True:
    
        tokens = list(word2freq.keys())
        freq = list(word2freq.values())

        for i in reversed(range(len(tokens))):
            if freq[i] <= 3:
                vocab.remove(tokens[i])


    # one hot encode vocabulary
        vocab_mapping = {}
        vocab_len = len(vocab)
        vocab = list(set(vocab))
        vocab.sort()
        print("Vocab length: ", vocab_len)
        for idx, word in tqdm(enumerate(vocab)):
            vocab_mapping[word] = idx
        print("Vocab length mapping: ", len(vocab_mapping))
        #save vocab to file
        f1 = open("./data/vocab.txt", "w")
        try:
            f1.write(json.dumps(vocab_mapping, sort_keys=True))
        finally:
            f1.close()
    
    else:
        f1 = open("./data/vocab.txt", "r")
        try:
            content = f1.read()
            vocab_mapping = json.loads(content)
        finally:
            f1.close()

    
    return vocab_mapping, sentences, labels


class Dataset(torch.utils.data.Dataset):
    '''
    Dataset class for the data
    '''
    def __init__(self, sentences, labels, vocab):
        self.sentences = sentences
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        assert idx < len(self)
        sentence = self.sentences[idx]
        label = self.labels[idx]
        sentence = [w.lower() for w in sentence if w.lower() in self.vocab.keys()]
        sentence = [self.vocab[w] for w in sentence]
        return sentence, label
    
    def iterate_w2v(self, batch_size):
        X = torch.zeros(batch_size, dtype=torch.int32)
        y = torch.zeros(batch_size, len(self.vocab))
        idx = 0
        for i in tqdm(range(0, len(self.sentences))):
            sentence = [w.lower() for w in self.sentences[i] if w.lower() in self.vocab.keys()]
            for i in range(len(sentence)):
                word = sentence[i]
                start = max(0, i - window_size)
                end = min(len(sentence) - 1, i + window_size)
                tot = end - start
                for j in range(start, end):
                    if j != i:
                        X[idx] = self.vocab[word]
                        y[idx, :] = one_hot(self.vocab[sentence[j]], len(self.vocab))
                        idx += 1
                    if idx < batch_size - 1 and (random.random() < 0.4):
                        # add negative samples
                        X[idx] = self.vocab[word]
                        y[idx, :] = torch.zeros(len(self.vocab))
                        idx += 1
                    if idx == batch_size - 1:
                        yield X, y
                        idx = 0

def generate_data(sentences, batch_size, vocab):
    # for each word in each sentence, generate a list of surrounding words

    X = []
    y = []
    for sentence in tqdm(sentences):
        sentence = [w.lower() for w in sentence if w.lower() in vocab.keys()]
        for i in range(len(sentence)):
            word = sentence[i]
            start = max(0, i - window_size)
            end = min(len(sentence) - 1, i + window_size)
            tot = end - start
            for j in range(start, end):
                if j != i:
                    X.append(vocab[word])
                    y.append(vocab[sentence[j]])
                if (random.random() < 0.4):
                    # add negative samples
                    X.append(vocab[word])
                    y.append(0)

            
            # print(word, [sentence[j] for j in range(max(0, i - window_size), min(len(sentence) - 1, i + window_size)) if j != i])
    print(len(X))
    print(len(y))
            

    X = np.array(X)
    y = np.array(y)

    X = X[: -(X.shape[0] % batch_size)]
    y = y[: -(y.shape[0] % batch_size)]
    return torch.tensor(X[..., np.newaxis].reshape(-1, batch_size)), torch.tensor(y[..., np.newaxis].reshape(-1, batch_size))


def get_k_closest(word, vocab, word2vec, k=10):
    word_embedding = word2vec[vocab[word]]
    print(word_embedding.shape)
    print(word2vec.shape)
    res = word2vec @ word_embedding[None, ...].T
    res = res.flatten()
    print(res.shape)

    best = list(torch.argsort(res, descending=True)[:k].numpy().flatten())
    worst = list(torch.argsort(res, descending=False)[:k].numpy().flatten())
    best_words = [v for v in vocab.items() if v[1] in best]
    worst_words = [v for v in vocab.items() if v[1] in worst]
    words_and_scores = [(w, res[i]) for w, i in best_words]
    worse_words_and_scores = [(w, res[i]) for w, i in worst_words]
    words_and_scores.sort(key=lambda x: x[1], reverse=True)
    worse_words_and_scores.sort(key=lambda x: x[1], reverse=False)
    return words_and_scores, worse_words_and_scores



def main():
    vocab, sentences, labels = generate_vocab_sentences_labels('./data/train.jsonl', save=GENERATE_VOCAB)
    print("Vocab Len:", len(vocab))

    # X, y = generate_data(sentences, batch_size, vocab)

    dataset = Dataset(sentences, labels, vocab)

    # train word2vec

    if TRAIN_WORD2VEC:
        model = train_word2vec(dataset, vocab, embedding_size, batch_size, num_epochs, learning_rate, lr_decay=lr_decay)
    else:
        model = Word2Vec(len(vocab), embedding_size)
        model.load_state_dict(torch.load('./model/word2vec.pth'))
        model.eval()

    best, worst = get_k_closest("love", vocab, model.embeding.weight, k=10)
    print(best)
    print(worst)


    cos = torch.nn.CosineSimilarity(dim=0).to("cuda")
    theta = cos(model.embeding.weight[vocab["king"]], model.embeding.weight[vocab["queen"]])
    print(theta)




if __name__ == "__main__":
    main()



# http://people.csail.mit.edu/mcollins/6864/slides/bikel.pdf