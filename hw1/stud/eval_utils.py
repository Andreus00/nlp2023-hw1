import torch
import numpy as np
import matplotlib.pyplot as plt

def cosine_similarity(embeddings, word):
    '''
    Computes the cosine similarity between two vectors
    '''
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    w = torch.einsum('ab, b -> ab', torch.ones_like(embeddings), word)
    return cos(embeddings, w)
     


def get_k_closest(word, vocab, embeddings, k=10):
    word_embedding = embeddings[vocab[word]]
    similarities = cosine_similarity(embeddings, embeddings[vocab[word]])
    best = list(torch.argsort(similarities)[-k:].numpy().flatten())
    worst = list(torch.argsort(similarities)[:k].numpy().flatten())
    best_words = [v for v in vocab.items() if v[1] in best]
    worst_words = [v for v in vocab.items() if v[1] in worst]
    words_and_scores = [(w, similarities[i]) for w, i in best_words]
    worse_words_and_scores = [(w, similarities[i]) for w, i in worst_words]
    words_and_scores.sort(key=lambda x: x[1], reverse=True)
    worse_words_and_scores.sort(key=lambda x: x[1], reverse=False)
    return words_and_scores, worse_words_and_scores



def plot_embeddings(vocab, embeddings):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    x_new = pca.fit_transform(embeddings)
    plt.scatter(x_new[:,0], x_new[:,1])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()
    


def plot_embeddings_close_to_word(vocab, embeddings, word, k=10):
    from sklearn.decomposition import PCA
    inv_vocab = {v: k for k, v in vocab.items()}
    pca = PCA(n_components=2)
    
    similarities = cosine_similarity(embeddings,  word)
        
    # print best and worst scores
    
    best_idxs = list(torch.argsort(similarities)[-k:].numpy().flatten())
    worst_idxs = list(torch.argsort(similarities)[:k].numpy().flatten())
    best_words = [v for v in vocab.items() if v[1] in best_idxs]
    worst_words = [v for v in vocab.items() if v[1] in worst_idxs]
    best_words_and_scores = [(w, similarities[i]) for w, i in best_words]
    worse_words_and_scores = [(w, similarities[i]) for w, i in worst_words]
    best_words_and_scores.sort(key=lambda x: x[1], reverse=True)
    worse_words_and_scores.sort(key=lambda x: x[1], reverse=False)
    print("-"*50)
    print()
    print("Best Scores:", best_words_and_scores)
    print()
    print("Worst Scores:", worse_words_and_scores)
    print()
    print("-"*50)
    
    
    # plot best words
    
    best = embeddings[best_idxs]
    x_new = pca.fit_transform(best)
    labels_of_x_new = [inv_vocab[i] for i in best_idxs]

    fig, ax = plt.subplots()
    ax.scatter(x_new[:,0], x_new[:,1])

    for i, txt in enumerate(labels_of_x_new):
        ax.annotate(txt, (x_new[i][0], x_new[i][1]))
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    
    plt.show()
    

