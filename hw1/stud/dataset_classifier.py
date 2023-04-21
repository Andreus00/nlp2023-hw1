import torch
from dataset_general import Dataset
from tqdm import tqdm
import config
import classes

class ClassifierDataset(torch.utils.data.IterableDataset, Dataset):

    def __init__(self, path, vocab_size, unk_token, window_size, get_key) -> None:
        super().__init__(path, vocab_size, unk_token, window_size)
        # self.word2id[config.PAD_TOKEN] = vocab_size
        # self.id2word[vocab_size] = config.PAD_TOKEN
        self.get_key = get_key
        
    
    def __iter__(self):
        sentences = self.data_words
        labels = self.labels
        i = 0
        while i < len(sentences):
            cur_sentence = sentences[i].copy()
            cur_labels = labels[i].copy()
            for k in range(len(cur_sentence)):
                cur_sentence[k] = self.get_key(cur_sentence[k])
                cur_labels[k] = classes.class2int[cur_labels[k]]
            yield {"inputs": cur_sentence, "targets": cur_labels}
            
            i += 1
    

    def get_stats(self):
        # count words
        import matplotlib.pyplot as plt
        import matplotlib
        import numpy as np
        import math

        # count how many times a given word occurs
        counter_list = [0 for _ in range(len(self.word2id) + 1)]
        for sentence in self.data_words:
            for word in sentence:
                if word in self.word2id:
                    counter_list[self.word2id[word]] += 1
        
        # count how many different words have the same number of occurrences
        num2occurrences = {}
        for num in counter_list:
            if num in num2occurrences:
                num2occurrences[num] += 1
            else:
                num2occurrences[num] = 1
        
        # get keys and values of the dictionary
        # keys are the number of occurrences
        # values are the number of words that have that number of occurrences
        keys = num2occurrences.keys()
        values = num2occurrences.values()
        print(keys)
        print(values)

        low_freq_words = 0
        mid_freq_words = 0
        high_freq_words = 0
        threshold_1 = 10
        threshold_2 = 100

        for i in range(len(counter_list)):
            if counter_list[i] <= threshold_1:
                low_freq_words += 1
            elif counter_list[i] <= threshold_2:
                mid_freq_words += 1
            else:
                high_freq_words += 1
        font = {'family' : 'normal',
                'size'   : 22}

        matplotlib.rc('font', **font)
        pps = plt.bar([1,2,3], [low_freq_words, mid_freq_words, high_freq_words], width=1, )
        plt.xticks([1,2,3], ["<10", "10-100", ">100"])
        plt.title("Words and their frequency")
        plt.xlabel("Frequency")
        plt.ylabel("Number of words")
        plt.show()

if __name__ == "__main__":
    from implementation import StudentModel
    model = model = StudentModel()
    train_dataset = ClassifierDataset(config.train_file, config.vocab_size, config.UNK_TOKEN, window_size=5, get_key=model.get_key)
    train_dataset.get_stats()
