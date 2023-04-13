from sklearn.decomposition import PCA
from torch import FloatTensor as FT
from torch import LongTensor as LT
import matplotlib.pyplot as plt
import torch.nn as nn
import collections
import numpy as np
import torch
import wget
import json
import os
import re

# for cute iteration bars (during training etc.)
from tqdm.auto import tqdm

torch.manual_seed(42)

class Word2VecDataset(torch.utils.data.IterableDataset):

    def __init__(self, txt_path, vocab_size, unk_token, window_size):
        """
        Args:
          txt_file (str): Path to the raw-text file.
          vocab_size (int): Maximum amount of words that we want to embed.
          unk_token (str): How will unknown words represented (e.g. 'UNK').
          window_size (int): Number of words to consider as context.
        """
        self.window_size = window_size
        # [[w1,s1, w2,s1, ..., w|s1|,s1], [w1,s2, w2,s2, ..., w|s2|,s2], ..., [w1,sn, ..., w|sn|,sn]]
        self.data_words = self.read_data(txt_path)
        self.build_vocabulary(vocab_size, unk_token)

    def __iter__(self):
        sentences = self.data_words
        for sentence in sentences:
            len_sentence = len(sentence)

            for input_idx in range(len_sentence):
                current_word = sentence[input_idx]
                # must be a word in the vocabulary
                if current_word in self.word2id and self.keep_word(current_word):
                    # left and right window indices
                    min_idx = max(0, input_idx - self.window_size)
                    max_idx = min(len_sentence, input_idx + self.window_size)

                    window_idxs = [x for x in range(min_idx, max_idx) if x != input_idx]
                    for target_idx in window_idxs:
                        # must be a word in the vocabulary
                        if sentence[target_idx] in self.word2id:
                            # index of target word in vocab
                            target = self.word2id[sentence[target_idx]]
                            # index of input word
                            current_word_id = self.word2id[current_word]
                            output_dict = {'targets':target, 'inputs':current_word_id}

                            yield output_dict

    def keep_word(self, word):
        '''Implements negative sampling and returns true if we can keep the occurrence as training instance.'''
        z = self.frequency[word] / self.tot_occurrences
        p_keep = np.sqrt(z / 10e-3) + 1
        p_keep *= 10e-3 / z # higher for less frequent instances
        return np.random.rand() < p_keep # toss a coin and compare it to p_keep to keep the word

    def read_data(self, txt_path):
        """Converts each line in the input file into a list of lists of tokenized words."""
        data = []
        total_words = 0
        # tot_lines = self.count_lines(txt_path)
        with open(txt_path) as f:
            for line in f:
                split = self.tokenize_line(line)
                if split:
                    data.append(split)
                    total_words += len(split)
        return data

    # "The pen is on the table" -> ["the, "pen", "is", "on", "the", "table"]
    def tokenize_line(self, line, pattern='\W'):
        """Tokenizes a single line."""
        return [word.lower() for word in re.split(pattern, line.lower()) if word]

    def build_vocabulary(self, vocab_size, unk_token):
        """Defines the vocabulary to be used. Builds a mapping (word, index) for
        each word in the vocabulary.

        Args:
          vocab_size (int): size of the vocabolary
          unk_token (str): token to associate with unknown words
        """
        counter_list = []
        # context is a list of tokens within a single sentence
        for context in self.data_words:
            counter_list.extend(context)
        counter = collections.Counter(counter_list)
        counter_len = len(counter)
        print("Number of distinct words: {}".format(counter_len))

        # consider only the (vocab size -1) most common words to build the vocab
        dictionary = {key: index for index, (key, _) in enumerate(counter.most_common(vocab_size - 1))}
        assert unk_token not in dictionary
        # all the other words are mapped to UNK
        dictionary[unk_token] = vocab_size - 1
        self.word2id = dictionary

        # dictionary with (word, frequency) pairs
        dict_counts = {x: counter[x] for x in dictionary if x is not unk_token}
        self.frequency = dict_counts
        self.tot_occurrences = sum(dict_counts[x] for x in dict_counts)

        print("Total occurrences of words in dictionary: {}".format(self.tot_occurrences))

        less_freq_word = min(dict_counts, key=counter.get)
        print("Less frequent word in dictionary appears {} times ({})".format(dict_counts[less_freq_word],
                                                                              less_freq_word))

        # index to word
        self.id2word = {value: key for key, value in dictionary.items()}

        # data is the text converted to indexes, as list of lists
        data = []
        # for each sentence
        for sentence in self.data_words:
            paragraph = []
            # for each word in the sentence
            for i in sentence:
                id_ = dictionary[i] if i in dictionary else dictionary[unk_token]
                if id_ == dictionary[unk_token]:
                    continue
                paragraph.append(id_)
            data.append(paragraph)
        # list of lists of indices, where each sentence is a list of indices, ignoring UNK
        self.data_idx = data


NEGATIVE_SAMPLING = False

# path to store checkpoints
PATH_OUTPUT_FOLDER = "./model/skipgram/"



VOCAB_SIZE = 10_000
UNK = 'UNK'

dataset = Word2VecDataset(train_data_path, VOCAB_SIZE, UNK, window_size=5)


class NEGLoss(nn.Module):
    '''Code from https://github.com/dthiagarajan/word2vec-pytorch/'''
    def __init__(self, idx_to_word, word_freqs, num_negative_samples=5):
        super(NEGLoss, self).__init__()
        # number of negative samples for which we will compute the loss
        self.num_negative_samples = num_negative_samples
        self.num_words = len(idx_to_word)
        # distribution of words 
        self.distr = nn.functional.normalize(torch.tensor([word_freqs[idx_to_word[i]]
                                                           for i in range(len(word_freqs))], dtype=float)
                                                  .pow(0.75),
                                             dim=0)

    def sample(self, num_samples, positives=None):
        # builds a matrix of weights for the loss function.
        # weights for negative samples are proportional to their frequency in the corpus.
        if positives is None:
            positives = []
        weights = torch.zeros((self.num_words, 1), device=device)
        for w in positives:
            weights[w] += 1.0
        for _ in range(num_samples):
            w = torch.multinomial(self.distr, 1)[0].to(device)
            while (w in positives):
                w = torch.multinomial(self.distr, 1)[0].to(device)
            weights[w] += 1.0
        return weights

    def forward(self, input, target):
        return nn.functional.nll_loss(input, target,
            self.sample(self.num_negative_samples, positives=target.data))


