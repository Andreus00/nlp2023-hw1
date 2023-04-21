import torch.nn as nn
import numpy as np
import torch
import config
from dataset_general import Dataset
import json
import re


# for cute iteration bars (during training etc.)
from tqdm.auto import tqdm

torch.manual_seed(42)

class Word2VecDataset(torch.utils.data.IterableDataset, Dataset):

    def __init__(self, path, vocab_size, unk_token, window_size):
        """
        Args:
          path (str): Path to the raw-text file.
          vocab_size (int): Maximum amount of words that we want to embed.
          unk_token (str): How will unknown words represented (e.g. 'UNK').
          window_size (int): Number of words to consider as context.
        """
        super().__init__(path, vocab_size, unk_token, window_size)
    

    def read_data(self, path):
        """Read sentences from train file and put them in a list.
        Remove stopwords and punctuation instead of just putting the UNK token.
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
        
        filt = lambda x: x not in stopwords and not re.match(r"(?=\S*['-])([a-zA-Z'-]+)", x) and not any([not c.isalnum() for c in x])

        for json_str in tqdm(json_list):
            result = json.loads(json_str)
            sentences.append([token.lower() for token in result['tokens'] if filt(token.lower())])
            
        return sentences, None

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
                            if config.ALGORITHM == config.USE_SKIPGRAM:
                                output_dict = {'targets':target, 'inputs':current_word_id}
                            elif config.ALGORITHM == config.USE_CBOW:
                                output_dict = {'targets':current_word_id, 'inputs':target}
                            else:
                                raise ValueError("Invalid algorithm: {}".format(config.ALGORITHM))
                            yield output_dict

    def keep_word(self, word):
        '''Implements negative sampling and returns true if we can keep the occurrence as training instance.'''
        z = self.frequency[word] / self.tot_occurrences
        p_keep = np.sqrt(z / 10e-3) + 1
        p_keep *= 10e-3 / z # higher for less frequent instances
        return np.random.rand() < p_keep # toss a coin and compare it to p_keep to keep the word


