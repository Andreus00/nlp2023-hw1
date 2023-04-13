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

# for cute iteration bars (during training etc.)
from tqdm.auto import tqdm

torch.manual_seed(42)

class Word2VecDataset(torch.utils.data.IterableDataset):

    def __init__(self, path, vocab_size, unk_token, window_size):
        """
        Args:
          path (str): Path to the raw-text file.
          vocab_size (int): Maximum amount of words that we want to embed.
          unk_token (str): How will unknown words represented (e.g. 'UNK').
          window_size (int): Number of words to consider as context.
        """
        self.window_size = window_size
        # [[w1,s1, w2,s1, ..., w|s1|,s1], [w1,s2, w2,s2, ..., w|s2|,s2], ..., [w1,sn, ..., w|sn|,sn]]
        self.data_words = self.read_data(path)
        self.build_vocabulary(vocab_size, unk_token)

    def __iter__(self):
        sentences = self.data_words
        pbar = tqdm(sentences, total=len(sentences))
        for sentence in pbar:
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

    def read_data(self, path):
        """Read sentences from train file and put them in a list."""
        gist_file = open("./data/gist_stopwords.txt", "r")
        stopwords = set()
        try:
            content = gist_file.read()
            stopwords = set(content.split(","))
        finally:
            gist_file.close()

        sentences = []

        with open(path, 'r') as json_file:
            json_list = list(json_file)

        for json_str in tqdm(json_list):
            result = json.loads(json_str)
            sentences.append([token.lower() for token in result['tokens'] if token not in stopwords])

        return sentences

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
