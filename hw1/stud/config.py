'''
Configureation file for the homework
'''
# trainig and validation files
train_file = 'data/train.jsonl'
validation_file = 'data/dev.jsonl'

# output files for the embedding models
OUTPUT_SKIPGRAM = "./model/skipgram/"
OUTPUT_CBOW = "./model/cbow/"
OUTPUT_GLOVE = "./model/glove/"

# embeddings selection
USE_CBOW = 0
USE_SKIPGRAM = 1
USE_GLOVE = 2
USE_W2V_GENSIM = 3
USE_FASTTEXT = 4


embedding_size = 300
window_size = 5
vocab_size = 20000 # only for cbow and skipgram

UNK_TOKEN = "UNK"
PAD_TOKEN = "PAD"

# classifier selection and configuration
ALGORITHM = USE_CBOW
MODEL_HANDLES_OOV = False
MODEL = 4
NORM_IN_RESBLOCK = True
UNFREEZE_EMB = False
UNFREEZE_EMB_EPOCH = 2
EMB_LR = 1e-5
USE_BIGRAMS = False
USE_POS_TAGGING = False

RESUME = None # "model/classifier/best_state.pt" # "model/classifier/double-bilstm-resblock-fasttext-no-bn.pt" #  "model/classifier/double-bilstm-resblock-glove-no-bn.pt"

# output for the custom embedding models (cbow and skipgram)
OUTPUT_W2V_PATH = OUTPUT_CBOW if ALGORITHM == USE_CBOW else OUTPUT_SKIPGRAM

# settings for the custom embedding models (cbow and skipgram)
NEGATIVE_SAMPLING = True
GENERATE_VOCAB = False

# output for the classifier
OUTPUT_CLASSIFIER = "./model/classifier/"

# hyperparameters
batch_size = 2048
learning_rate = 1e-4
weight_decay = 1e-4
lr_decay=0.9
num_epochs = 50


# switch between training the classifier and the embedding models
TRAIN_CLASSIFIER = False

# deprecated
TRAIN = False
EVALUATE = True

# wandb
WANDB = False

# device selection
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

