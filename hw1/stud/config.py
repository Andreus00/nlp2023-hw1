train_file = 'data/train.jsonl'

OUTPUT_SKIPGRAM = "./model/skipgram/"
OUTPUT_CBOW = "./model/cbow/"

USE_CBOW = 0
USE_SKIPGRAM = 1

ALGORITHM = USE_SKIPGRAM

OUTPUT_PATH = OUTPUT_CBOW if ALGORITHM == USE_CBOW else OUTPUT_SKIPGRAM

batch_size = 256
learning_rate = 1e-2
weight_decay = 0
lr_decay=1
num_epochs = 10
embedding_size = 300
window_size = 5
vocab_size = 27986


UNK_TOKEN = "UNK"
TESTING = True
NEGATIVE_SAMPLING = True

GENERATE_VOCAB = False
PLOT_EMBEDDINGS = True
RESUME = False
TRAIN = True
EVALUATE = True


import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
