train_file = 'data/train.jsonl'
validation_file = 'data/dev.jsonl'

OUTPUT_SKIPGRAM = "./model/skipgram/"
OUTPUT_CBOW = "./model/cbow/"
OUTPUT_GLOVE = "./model/glove/"

USE_CBOW = 0
USE_SKIPGRAM = 1
USE_GLOVE = 2
USE_W2V_GENSIM = 3
USE_FASTTEXT = 4

ALGORITHM = USE_FASTTEXT
MODEL_HANDLES_OOV = True

MODEL = 3


OUTPUT_W2V_PATH = OUTPUT_CBOW if ALGORITHM == USE_CBOW else OUTPUT_SKIPGRAM

OUTPUT_CLASSIFIER = "./model/classifier/"

batch_size = 128
learning_rate = 1e-4
weight_decay = 1e-5
lr_decay=1
num_epochs = 30
embedding_size = 300
window_size = 5
vocab_size = 20000 #  27986


UNK_TOKEN = "UNK"
PAD_TOKEN = "PAD"
TESTING = True
NEGATIVE_SAMPLING = True

GENERATE_VOCAB = False
PLOT_EMBEDDINGS = True
RESUME = None # "model/classifier/best_state.pt" # "model/classifier/double-bilstm-resblock-fasttext-no-bn.pt" #  "model/classifier/double-bilstm-resblock-glove-no-bn.pt"
TRAIN = True
EVALUATE = True
TRAIN_CLASSIFIER = True
WANDB = True
UNFREEZE_EMB = False
NORM_IN_RESBLOCK = True


import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

