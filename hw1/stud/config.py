train_file = 'data/train.jsonl'

OUTPUT_SKIPGRAM = "./model/skipgram/"
OUTPUT_CBOW = "./model/cbow/"

USE_CBOW = 0
USE_SKIPGRAM = 1

ALGORITHM = USE_CBOW

OUTPUT_PATH = OUTPUT_CBOW if ALGORITHM == USE_CBOW else OUTPUT_SKIPGRAM

batch_size = 256
learning_rate = 3e-3
weight_decay = 1e-5
lr_decay=1
num_epochs = 100
embedding_size = 300
window_size = 5
vocab_size = 10000


UNK_TOKEN = "UNK"
TESTING = True
NEGATIVE_SAMPLING = True

GENERATE_VOCAB = False
PLOT_EMBEDDINGS = True
RESUME = True
TRAIN = True


import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
