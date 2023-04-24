import gensim
from gensim import downloader as api
from gensim.models import FastText
import numpy as np
import torch
import config
import json

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
from gensim.test.utils import datapath
import io
from typing import Dict, List


# def LoadFastText():
#     input_file = io.open('model/fasttext/wiki-news-300d-1M-subword.vec', 'r', encoding='utf-8', newline='\n', errors='ignore')
#     no_of_words, vector_size = map(int, input_file.readline().split())
#     word_to_vector: Dict[str, List[float]] = dict()
#     for i, line in enumerate(input_file):
#         tokens = line.rstrip().split(' ')
#         word = tokens[0]
#         vector = list(map(float, tokens[1:]))
#         assert len(vector) == vector_size
#         word_to_vector[word] = vector
#     return word_to_vector

# model = LoadFastText()
# print(model["hi"])


# TODO: implement this embedding model -> fasttext-wiki-news-subwords-300

# corpus = api.load('text8')
# model = FastText(vector_size=300, window=5, min_count=1)
# model.build_vocab(corpus_iterable=corpus)
# model.train(corpus_iterable=corpus, total_words=model.corpus_total_words, epochs=10)  # train
# model.save('model/fasttext/fasttext-wiki-news-subwords-300.bin')

# print(emb_model)
# emb_model = gensim.models.KeyedVectors.load_word2vec_format('model/glove/glove-wiki-gigaword-300.gz')
# emb_model.vectors = np.append(emb_model.vectors, torch.zeros(1, 300), axis=0)
# emb_model.key_to_index[config.UNK_TOKEN] = emb_model.vectors.shape[0] - 1
# emb_model.index_to_key.append(config.UNK_TOKEN)
# weights = torch.FloatTensor(emb_model.vectors)
# vocab = emb_model.key_to_index
# embedding = torch.nn.Embedding.from_pretrained(weights)
# torch.save(embedding, 'model/w2v/word2vec-google-news-300.pt')
# with open('model/w2v/word2vec-google-news-300.json', 'w') as f:
#     json.dump(vocab, f)



# dataset = Word2VecDataset(config.train_file, config.vocab_size, config.UNK_TOKEN, window_size=5)
# emb_model = CBOW(config.vocab_size, embedding_dim=config.embedding_size, id2word=dataset.id2word,
#             word_counts=dataset.frequency, NEG_SAMPLING=config.NEGATIVE_SAMPLING)
# emb_model.load_state_dict(torch.load('model/cbow/state_54.pt'))
# embedding = torch.nn.Embedding.from_pretrained(emb_model.get_embeddings())
# torch.save(embedding, "model/cbow/embeddings.pt")
# print("Embeddings saved.")
# # print(dataset.word2id)
# with open("model/cbow/vocab.json", "a+") as f:
#     json.dump(dataset.word2id, f)
# print("Vocab saved.")
# import time
# time.sleep(10)
# # print()
# print("Trying to load embeddings:")
# embedding = torch.load("model/cbow/embeddings.pt")
# print("Embeddings loaded. Loading vocab:")
# vocab = None
# with open("model/cbow/vocab.json", "r") as f:
#     vocab = json.load(f)
# print(vocab)
# print("Vocab loaded. Testing:")
# print("Cat: ", embedding(torch.tensor([vocab["cat"]])))
