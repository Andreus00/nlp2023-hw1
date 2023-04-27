import config
import torch
import os
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np

if config.WANDB:
    import wandb
    wandb.init(
        # set the wandb project where this run will be logged
        project="nlp_homework",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": config.learning_rate,
        "architecture": "LSTM-512-2layers",
        "epochs": 30,
        }
    )

class Trainer():
    '''
    This class trains a model on a given dataset.

    It can be used for both classification and embeddings, based on the configuration.
    '''
    def __init__(self, model, optimizer):

        self.device = config.device

        self.model = model
        self.optimizer = optimizer

        # starts requires_grad for all layers
        self.model.train()  # we are using this model for training (some layers have different behaviours in train and eval mode)
        self.model.to(self.device)  # move model to GPU if available

    def train(self, train_dataset, output_folder, epochs=1, num_batches=1, validation_dataset=None):

        train_loss = 0.0
        best_val_loss = np.inf
        for epoch in range(epochs):
            epoch_loss = 0.0
            len_train = 0
            cumulative_accuracy, cumulative_precision, cumulative_f1, cumulative_recall = 0.0, 0.0, 0.0, 0.0
            pbar = tqdm(enumerate(train_dataset), total=num_batches, desc="Epoch {}".format(epoch))

            for step, sample in pbar:
                if config.TRAIN_CLASSIFIER:
                    if config.USE_BIGRAMS and not config.USE_POS_TAGGING:
                        inputs = sample[0].to(self.device)
                        original_lengths = sample[1]
                        targets = sample[2].to(self.device)
                        bigrams = sample[3].to(self.device)
                    elif config.USE_BIGRAMS and config.USE_POS_TAGGING:
                        inputs = sample[0].to(self.device)
                        original_lengths = sample[1]
                        targets = sample[2].to(self.device)
                        bigrams = sample[3].to(self.device)
                        pos = sample[4].to(self.device)
                    else:
                        inputs = sample[0].to(self.device)
                        original_lengths = sample[1]
                        targets = sample[2].to(self.device)
                else:
                    targets = sample["targets"].to(self.device)
                    inputs = sample["inputs"].to(self.device)

                if config.TRAIN_CLASSIFIER:
                    if config.USE_BIGRAMS and not config.USE_POS_TAGGING:
                        X = (inputs, bigrams)
                    elif config.USE_BIGRAMS and config.USE_POS_TAGGING:
                        X = (inputs, bigrams, pos)
                    else:
                        X = inputs
                else:
                    X = torch.zeros((inputs.shape[0], config.vocab_size), device=self.device)
                    for idx in range(inputs.shape[0]):
                        X[idx, inputs[idx]] = 1
                
                output_distribution = self.model(X)

                if config.TRAIN_CLASSIFIER:
                    output_distribution = torch.nn.utils.rnn.pack_padded_sequence(output_distribution, original_lengths, batch_first=True, enforce_sorted=False)[0]
                    targets = torch.nn.utils.rnn.pack_padded_sequence(targets, original_lengths, batch_first=True, enforce_sorted=False)[0]
                    
                loss = self.model.loss_function(output_distribution, targets)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()
                len_train += 1
                # calculate metrics

                targets = targets.detach().cpu().numpy()
                output_distribution = output_distribution.detach().cpu().numpy().argmax(axis=1)
                remove_idx = np.where(targets == -100)
                targets = np.delete(targets, remove_idx)
                output_distribution = np.delete(output_distribution, remove_idx)

                accuracy = accuracy_score(targets, output_distribution)
                precision = precision_score(targets, output_distribution, average='macro', zero_division=True)
                f1 = f1_score(targets, output_distribution, average='macro', zero_division=True)
                recall = recall_score(targets, output_distribution, average='macro', zero_division=True)
                pbar.set_postfix(loss=loss.item(), f1="{:0.4f}".format(f1), accuracy="{:0.4f}".format(accuracy), precision="{:0.4f}".format(precision), recall="{:0.4f}".format(recall))
                cumulative_accuracy += accuracy
                cumulative_precision += precision
                cumulative_f1 += f1
                cumulative_recall += recall
                del targets, inputs, output_distribution
            avg_epoch_loss = epoch_loss / len_train
            avg_epoch_accuracy = cumulative_accuracy / len_train
            avg_epoch_precision = cumulative_precision / len_train
            avg_epoch_f1 = cumulative_f1 / len_train
            avg_epoch_recall = cumulative_recall / len_train

            print('Epoch: {} avg loss = {:0.10f},  avg f1 {:0.4f} avg accuracy {:0.4f} avg precision {:0.4f}  avg recall {:0.4f}'.format(epoch, avg_epoch_loss, avg_epoch_f1, avg_epoch_accuracy, avg_epoch_precision, avg_epoch_recall))
            # with open(os.path.join(output_folder, 'train_metrics.txt'), 'a') as f:
            #             f.write('Validaton Epoch: {} avg loss = {:0.10f},  avg f1 {:0.4f} avg accuracy {:0.4f} avg precision {:0.4f}  avg recall {:0.4f}\n'.format(epoch, avg_epoch_loss, avg_epoch_f1, avg_epoch_accuracy, avg_epoch_precision, avg_epoch_recall))

            train_loss += avg_epoch_loss

            if not config.TRAIN_CLASSIFIER and config.WANDB:
                    wandb.log({ 'train_loss': avg_epoch_loss, 'train_f1': avg_epoch_f1, 'train_accuracy': avg_epoch_accuracy, 'train_precision': avg_epoch_precision, 'train_recall': avg_epoch_recall})


            if validation_dataset is not None:
                print("Validation")
                with torch.no_grad():
                    val_loss = 0.0
                    len_val = 0
                    cumulative_accuracy, cumulative_precision, cumulative_f1, cumulative_recall = 0.0, 0.0, 0.0, 0.0
                    for sample in validation_dataset:
                        if config.TRAIN_CLASSIFIER:
                            if config.USE_BIGRAMS and not config.USE_POS_TAGGING:
                                inputs = sample[0].to(self.device)
                                original_lengths = sample[1]
                                targets = sample[2].to(self.device)
                                bigrams = sample[3].to(self.device)
                            elif config.USE_BIGRAMS and config.USE_POS_TAGGING:
                                inputs = sample[0].to(self.device)
                                original_lengths = sample[1]
                                targets = sample[2].to(self.device)
                                bigrams = sample[3].to(self.device)
                                pos = sample[4].to(self.device)
                            else:
                                inputs = sample[0].to(self.device)
                                original_lengths = sample[1]
                                targets = sample[2].to(self.device)
                        else:
                            targets = sample[1].to(self.device)
                        inputs = sample[0].to(self.device)

                        if config.TRAIN_CLASSIFIER:
                            if config.USE_BIGRAMS and not config.USE_POS_TAGGING:
                                X = (inputs, bigrams)
                            elif config.USE_BIGRAMS and config.USE_POS_TAGGING:
                                X = (inputs, bigrams, pos)
                            else:
                                X = inputs
                        else:
                            X = torch.zeros((inputs.shape[0], config.vocab_size), device=self.device)
                            for word in range(inputs.shape[0]):
                                X[word, inputs[word]] = 1

                        output_distribution = self.model(X)

                        if config.TRAIN_CLASSIFIER:
                            output_distribution = torch.nn.utils.rnn.pack_padded_sequence(output_distribution, original_lengths, batch_first=True, enforce_sorted=False)[0]
                            targets = torch.nn.utils.rnn.pack_padded_sequence(targets, original_lengths, batch_first=True, enforce_sorted=False)[0]
                        
                        loss = self.model.loss_function(output_distribution, targets)

                        val_loss += loss.item()
                        len_val += 1

                        targets = targets.detach().cpu().numpy()
                        output_distribution = output_distribution.detach().cpu().numpy().argmax(axis=1)

                        remove_idx = np.where(targets == -100)
                        targets = np.delete(targets, remove_idx)
                        output_distribution = np.delete(output_distribution, remove_idx)

                        accuracy = accuracy_score(targets, output_distribution)
                        precision = precision_score(targets, output_distribution, average='macro', zero_division=True)
                        f1 = f1_score(targets, output_distribution, average='macro', zero_division=True)
                        recall = recall_score(targets, output_distribution, average='macro', zero_division=True)
                        cumulative_accuracy += accuracy
                        cumulative_precision += precision
                        cumulative_f1 += f1
                        cumulative_recall += recall
                        del targets, inputs, output_distribution
                    avg_val_loss = val_loss / len_val
                    avg_val_accuracy = cumulative_accuracy / len_val
                    avg_val_precision = cumulative_precision / len_val
                    avg_val_f1 = cumulative_f1 / len_val
                    avg_val_recall = cumulative_recall / len_val
                    print('ValidationEpoch: {} avg loss = {:0.10f},  avg f1 {:0.4f} avg accuracy {:0.4f} avg precision {:0.4f}  avg recall {:0.4f}'.format(epoch, avg_val_loss, avg_val_f1, avg_val_accuracy, avg_val_precision, avg_val_recall))


                if config.WANDB:
                    wandb.log({"val-acc": avg_val_accuracy, "val-loss": avg_val_loss, "val-f1": avg_val_f1, "val-precision": avg_val_precision, "val-recall": avg_val_recall, 'train_loss': avg_epoch_loss, 'train_f1': avg_epoch_f1, 'train_accuracy': avg_epoch_accuracy, 'train_precision': avg_epoch_precision, 'train_recall': avg_epoch_recall})


                if best_val_loss > avg_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(self.model.state_dict(),
                        os.path.join(output_folder, 'best_state.pt'))
                    print('------- Model saved at epoch {}  -------'.format(epoch))
            else:
                torch.save(self.model.state_dict(),
                        os.path.join(output_folder, 'state_{}.pt'.format(epoch)))
            
            if config.UNFREEZE_EMB and epoch == config.UNFREEZE_EMB_EPOCH:
                self.model.unfreeze_embeddings()
                print("Unfreeze embeddings")
                

        avg_epoch_loss = train_loss / epochs
        if config.WANDB:
            wandb.finish()
        return avg_epoch_loss
    