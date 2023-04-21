from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import os
import config
from dataset_classifier import ClassifierDataset
from implementation import StudentModel
from tqdm.auto import tqdm
from eval_utils import plot_embeddings_close_to_word
from trainer import Trainer
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import classes

torch.manual_seed(42)

def collate_fn(data):
	'''  
	We should build a custom collate_fn rather than using default collate_fn,
	as the size of every sentence is different and merging sequences (including padding) 
	is not supported in default. 
	Args:
		data: list of tuple (training sequence, label)
	Return:
	padded_seq - Padded Sequence, tensor of shape (batch_size, padded_length)
	length - Original length of each sequence(without padding), tensor of shape(batch_size)
	label - tensor of shape (batch_size)
	'''
	#sorting is important for usage pack padded sequence (used in model). It should be in decreasing order.
	# save original length of each sequence
	length = [len(x["inputs"]) for x in data]
	inputs = [torch.tensor(x["inputs"]) for x in data]
	targets = [torch.tensor(x["targets"]) for x in data]

	if len(inputs) < config.batch_size:
		# add the remaining samples to the batch. These samples will be ignored in the loss function
		for i in range(config.batch_size - len(inputs)):
			inputs.append(torch.tensor([config.vocab_size - 1]))
			targets.append(torch.tensor([-100]))
			length.append(1)
	# pad inputs
	padded_seq = pad_sequence(inputs, batch_first=True, padding_value=config.vocab_size - 1)
	padded_targets = pad_sequence(targets, batch_first=True, padding_value=-100)
	return padded_seq, torch.tensor(length), padded_targets




## load the dataset and the model

model = StudentModel().to(config.device)
train_dataset = ClassifierDataset(config.train_file, config.vocab_size, config.UNK_TOKEN, window_size=5, get_key=model.get_key)
validation_dataset = ClassifierDataset(config.validation_file, config.vocab_size, config.UNK_TOKEN, window_size=5, get_key=model.get_key)
if config.ALGORITHM == config.USE_GLOVE:
	validation_dataset.word2id = model.vocab
	validation_dataset.id2word = {v: k for k, v in model.vocab.items()}
	train_dataset.word2id = model.vocab
	train_dataset.id2word = {v: k for k, v in model.vocab.items()}
	config.vocab_size = len(model.vocab)
else:
	validation_dataset.word2id = train_dataset.word2id
	validation_dataset.id2word = train_dataset.id2word
# define an optimizer to update the parameters

optimizer = torch.optim.Adagrad(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)


## Train the model

if config.TRAIN:
	trainer = Trainer(model, optimizer)
	dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=collate_fn) #it batches data for us
	val_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=config.batch_size, collate_fn=collate_fn) #it batches data for us
	avg_loss = trainer.train(dataloader, config.OUTPUT_CLASSIFIER, epochs=config.num_epochs, num_batches=len(train_dataset.data_words)//config.batch_size, validation_dataset=val_dataloader)


