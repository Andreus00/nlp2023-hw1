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

'''
Script that trains a classifier.
'''


def collate_fn(data):
	'''
	I wrote a custom collate function to handle the padding of the sequences.
	'''
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

	returns = [padded_seq, torch.tensor(length), padded_targets]

	# add bigrams to the output if required
	if config.USE_BIGRAMS:
		bigrams = [torch.tensor(x["bigrams"]) for x in data]
		if len(bigrams) < config.batch_size:
			for i in range(config.batch_size - len(bigrams)):
				bigrams.append(torch.tensor([config.vocab_size - 1]))
		padded_bigrams = pad_sequence(bigrams, batch_first=True, padding_value=config.vocab_size - 1)
		returns.append(padded_bigrams)
	
	if config.USE_POS_TAGGING:
		pos = [torch.tensor(x["pos"]) for x in data]
		if len(pos) < config.batch_size:
			for i in range(config.batch_size - len(pos)):
				pos.append(torch.tensor([classes.pos2int["PAD"]]))
		padded_pos = pad_sequence(pos, batch_first=True, padding_value=classes.pos2int["PAD"])
		returns.append(padded_pos)

	return returns




## load the dataset and the model

model = StudentModel().to(config.device)
train_dataset = ClassifierDataset(config.train_file, config.vocab_size, config.UNK_TOKEN, window_size=5, get_key=model.get_key)
validation_dataset = ClassifierDataset(config.validation_file, config.vocab_size, config.UNK_TOKEN, window_size=5, get_key=model.get_key)

# this part is no longer needed since the dataset now handles the conversion
# from word to id thaks to the get_key function
if config.ALGORITHM == config.USE_GLOVE or config.ALGORITHM == config.USE_W2V_GENSIM:
	validation_dataset.word2id = model.vocab
	validation_dataset.id2word = {v: k for k, v in model.vocab.items()}
	train_dataset.word2id = model.vocab
	train_dataset.id2word = {v: k for k, v in model.vocab.items()}
	config.vocab_size = len(model.vocab)
elif config.ALGORITHM == config.USE_FASTTEXT:
	validation_dataset.word2id = model.embedding.key_to_index
	validation_dataset.id2word = {v: k for k, v in model.embedding.key_to_index.items()}
	train_dataset.word2id = model.embedding.key_to_index
	train_dataset.id2word = {v: k for k, v in model.embedding.key_to_index.items()}
	config.vocab_size = len(model.embedding.key_to_index)
else:
	validation_dataset.word2id = train_dataset.word2id
	validation_dataset.id2word = train_dataset.id2word


# define an optimizer to update the parameters
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)


## Train the model
if config.TRAIN:
	trainer = Trainer(model, optimizer)
	dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=collate_fn) #it batches data for us
	val_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=config.batch_size, collate_fn=collate_fn) #it batches data for us
	avg_loss = trainer.train(dataloader, config.OUTPUT_CLASSIFIER, epochs=config.num_epochs, num_batches=len(train_dataset.data_words)//config.batch_size, validation_dataset=val_dataloader)


