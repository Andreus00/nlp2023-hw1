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
from cbow import CBOW, train_cbow
import config


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


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
                start = max(0, i - config.window_size)
                end = min(len(sentence) - 1, i + config.window_size)
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
    
    def iterate_cbow(self, batch_size):
        X = torch.zeros(batch_size, dtype=torch.int32)
        y = torch.zeros(batch_size, len(self.vocab))
        idx = 0
        for i in tqdm(range(0, len(self.sentences))):
            sentence = [w.lower() for w in self.sentences[i] if w.lower() in self.vocab.keys()]
            for i in range(len(sentence)):
                word = sentence[i]
                start = max(0, i - config.window_size)
                end = min(len(sentence) - 1, i + config.window_size)
                tot = end - start
                for j in range(start, end):
                    if j != i:
                        X[idx] = self.vocab[sentence[j]]
                        y[idx, :] = one_hot(self.vocab[word], len(self.vocab))
                        idx += 1
                    if idx == batch_size - 1:
                        yield X, y
                        idx = 0

    def generate_data(self, batch_size):
        # for each word in each sentence, generate a list of surrounding words

        X = []
        y = []
        for sentence in tqdm(self.sentences):
            sentence = [w.lower() for w in sentence if w.lower() in self.vocab.keys()]
            for i in range(len(sentence)):
                word = sentence[i]
                start = max(0, i - config.window_size)
                end = min(len(sentence) - 1, i + config.window_size)
                tot = end - start
                for j in range(start, end):
                    if j != i:
                        X.append(self.vocab[sentence[j]])
                        y.append(self.vocab[word])

                
                # print(word, [sentence[j] for j in range(max(0, i - window_size), min(len(sentence) - 1, i + window_size)) if j != i])
        print(len(X))
        print(len(y))
                

        X = np.array(X)
        y = np.array(y)

        X = X[: -(X.shape[0] % batch_size)]
        y = y[: -(y.shape[0] % batch_size)]
        return torch.tensor(X[..., np.newaxis].reshape(-1, batch_size)), torch.tensor(y[..., np.newaxis].reshape(-1, batch_size))


def cosine_similarity(embeddings, word):
    '''
    Computes the cosine similarity between two vectors
    '''
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    w = torch.einsum('ab, b -> ab', torch.ones_like(embeddings), word)
    return cos(embeddings, w)
     


def get_k_closest(word, vocab, embeddings, k=10):
    word_embedding = embeddings[vocab[word]]
    similarities = cosine_similarity(embeddings, embeddings[vocab[word]])
    best = list(torch.argsort(similarities)[-k:].numpy().flatten())
    worst = list(torch.argsort(similarities)[:k].numpy().flatten())
    best_words = [v for v in vocab.items() if v[1] in best]
    worst_words = [v for v in vocab.items() if v[1] in worst]
    words_and_scores = [(w, similarities[i]) for w, i in best_words]
    worse_words_and_scores = [(w, similarities[i]) for w, i in worst_words]
    words_and_scores.sort(key=lambda x: x[1], reverse=True)
    worse_words_and_scores.sort(key=lambda x: x[1], reverse=False)
    return words_and_scores, worse_words_and_scores



def plot_embeddings(vocab, embeddings):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    x_new = pca.fit_transform(embeddings)
    plt.scatter(x_new[:,0], x_new[:,1])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()
    


def plot_embeddings_close_to_word(vocab, embeddings, word, k=10):
    from sklearn.decomposition import PCA
    inv_vocab = {v: k for k, v in vocab.items()}
    pca = PCA(n_components=2)
    
    similarities = cosine_similarity(embeddings,  word)
    
    # print best and worst scores
    
    best_idxs = list(torch.argsort(similarities)[-k:].numpy().flatten())
    worst_idxs = list(torch.argsort(similarities)[:k].numpy().flatten())
    best_words = [v for v in vocab.items() if v[1] in best_idxs]
    worst_words = [v for v in vocab.items() if v[1] in worst_idxs]
    best_words_and_scores = [(w, similarities[i]) for w, i in best_words]
    worse_words_and_scores = [(w, similarities[i]) for w, i in worst_words]
    best_words_and_scores.sort(key=lambda x: x[1], reverse=True)
    worse_words_and_scores.sort(key=lambda x: x[1], reverse=False)
    print("Best Scores:", best_words_and_scores)
    print()
    print("Worst Scores:", worse_words_and_scores)
    
    
    # plot best words
    
    best = embeddings[best_idxs]
    x_new = pca.fit_transform(best)
    labels_of_x_new = [inv_vocab[i] for i in best_idxs]

    fig, ax = plt.subplots()
    ax.scatter(x_new[:,0], x_new[:,1])

    for i, txt in enumerate(labels_of_x_new):
        ax.annotate(txt, (x_new[i][0], x_new[i][1]))
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    
    plt.show()
    



def main():
    vocab, sentences, labels = generate_vocab_sentences_labels('./data/train.jsonl', save=config.GENERATE_VOCAB)
    print("Vocab Len:", len(vocab))

    dataset = Dataset(sentences, labels, vocab)

    # train cbow

    if config.TRAIN_CBOW:
        model = train_cbow(dataset, vocab, config.embedding_size, config.batch_size, config.num_epochs, config.learning_rate, lr_decay=config.lr_decay)
    else:
        model = CBOW(len(vocab), config.embedding_size)
        model.load_state_dict(torch.load('./model/cbow.pth'))
        model.eval()
        
    CBOW_WEIGHTS = model.lin_layer.weight
        
    TEST_WORD = "april"
    
    if config.PLOT_EMBEDDINGS:
        plot_embeddings_close_to_word(vocab, CBOW_WEIGHTS.detach(), CBOW_WEIGHTS[vocab[TEST_WORD]], k=30)


if __name__ == "__main__":
    main()



# http://people.csail.mit.edu/mcollins/6864/slides/bikel.pdf