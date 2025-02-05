import torch.nn as nn
import numpy as np
import torch
import config
from tqdm import tqdm
import json
import re
import os
import collections
import nltk
nltk.download('averaged_perceptron_tagger')
if config.USE_BIGRAMS:
    from gensim.models import Phrases
    from gensim.models.phrases import Phraser


class Dataset:
    """
    General dataset class.

    It implements initialization, vocabulary building and data loading.
    """

    def __init__(self, path, vocab_size, unk_token, window_size, ) -> None:
        super().__init__()
        self.window_size = window_size
        # [[w1,s1, w2,s1, ..., w|s1|,s1], [w1,s2, w2,s2, ..., w|s2|,s2], ..., [w1,sn, ..., w|sn|,sn]]
        if config.USE_BIGRAMS and not config.USE_POS_TAGGING:
            self.data_words, self.data_bigrams, self.labels = self.read_data(path)
        elif config.USE_BIGRAMS and config.USE_POS_TAGGING:
            self.data_words, self.data_bigrams, self.data_pos, self.labels = self.read_data(path)
        else:
            self.data_words, self.labels = self.read_data(path)
        self.build_vocabulary(vocab_size, unk_token)

    def read_data(self, path):
        """
        Read sentences from train file and put them in a list.
        """

        gist_file = open("./data/gist_stopwords.txt", "r")
        stopwords = set()
        try:
            content = gist_file.read()
            stopwords = set(content.split(","))
        finally:
            gist_file.close()

        sentences = []
        labels = []

        with open(path, 'r') as json_file:
            json_list = list(json_file)
        
        filt = lambda x: True #  x not in stopwords and not re.match(r"(?=\S*['-])([a-zA-Z'-]+)", x) and not any([not c.isalnum() for c in x])

        pos_tags = []

        for json_str in tqdm(json_list):
            result = json.loads(json_str)
            sentences.append([token.lower()  if filt(token.lower()) else config.UNK_TOKEN for token in result['tokens']])
            labels.append(result['labels'])
            if config.USE_POS_TAGGING:
                pos_tags.append([token[1] for token in nltk.pos_tag(result['tokens'], tagset='universal')])

        returns = [sentences, labels]

        if config.USE_POS_TAGGING:
            returns.insert(1, pos_tags)
            print("POS TAGS: ", pos_tags[0:10])
        
        if config.USE_BIGRAMS:
            bigram = Phrases(sentences, min_count=1, threshold=1)
            bigram_phraser = Phraser(bigram)
            bigrams = [bigram_phraser[sentence] for sentence in sentences]
            # i have to add padding after each bigram. Not every word is a bigram, so the length of the sentence changes
            for i in range(len(bigrams)):
                bigrams[i] += [config.PAD_TOKEN] * (len(sentences[i]) - len(bigrams[i]))
            
            returns.insert(1, bigrams)
        
        return returns

    def build_vocabulary(self, vocab_size, unk_token):
        """
        Defines the vocabulary to be used. Builds a mapping (word, index) for
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
        # assert unk_token not in dictionary
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