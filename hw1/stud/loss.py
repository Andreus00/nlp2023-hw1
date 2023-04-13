import torch.nn as nn
import torch
import config

class NEGLoss(nn.Module):
    '''Code from https://github.com/dthiagarajan/word2vec-pytorch/'''
    def __init__(self, idx_to_word, word_freqs, num_negative_samples=5):
        super(NEGLoss, self).__init__()
        # number of negative samples for which we will compute the loss
        self.num_negative_samples = num_negative_samples
        self.num_words = len(idx_to_word)
        # distribution of words 
        self.distr = nn.functional.normalize(torch.tensor([word_freqs[idx_to_word[i]]
                                                           for i in range(len(word_freqs))], dtype=float)
                                                  .pow(0.75),
                                             dim=0)

    def sample(self, num_samples, positives=None):
        # builds a matrix of weights for the loss function.
        # weights for negative samples are proportional to their frequency in the corpus.
        device = config.device
        if positives is None:
            positives = []
        weights = torch.zeros((self.num_words, 1), device=device)
        for w in positives:
            weights[w] += 1.0
        for _ in range(num_samples):
            w = torch.multinomial(self.distr, 1)[0].to(device)
            while (w in positives):
                w = torch.multinomial(self.distr, 1)[0].to(device)
            weights[w] += 1.0
        return weights.flatten()

    def forward(self, input, target):
        return nn.functional.nll_loss(input, target,
            self.sample(self.num_negative_samples, positives=target.data))
