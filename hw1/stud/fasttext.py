# my fasttext wrapper
import numpy as np
import io
import config
# from gensim.models import FastText
import json
import torch
# from gensim.models.fasttext_inner import ft_hash_bytes


def compute_ngrams_bytes(word, minn, maxn):
    """Compute ngrams for a given word.

    Parameters
    ----------
    word : str
        The word to compute ngrams for.
    minn : int
        Minimum ngram length
    maxn : int
        Maximum ngram length

    Returns
    -------
    list of bytes
        The ngrams of the word.

    """
    utf8_word = ('<%s>' % word).encode("utf-8")
    bytez = utf8_word
    num_bytes = len(utf8_word)
    _MB_MASK = 0xC0
    _MB_START = 0x80
    ngrams = []
    for i in range(num_bytes):
        if bytez[i] & _MB_MASK == _MB_START:
            continue

        j, n = i, 1
        while j < num_bytes and n <= maxn:
            j += 1
            while j < num_bytes and (bytez[j] & _MB_MASK) == _MB_START:
                j += 1
            if n >= minn and not (n == 1 and (i == 0 or j == num_bytes)):
                ngram = bytes(bytez[i:j])
                ngrams.append(ngram)
            n += 1
    return ngrams

def ft_hash_bytes_my(bytes):
    """Hash a byte string.

    Parameters
    ----------
    bytes : bytes
        The bytes to hash.

    Returns
    -------
    int
        The hash of the bytes.

    """
    h = np.uint32(2166136261)

    for b in bytes:
        h = h ^ np.uint32(np.int8(b))
        h = h * np.uint32(16777619)
    return h


def ft_ngram_hashes(word, minn, maxn, num_buckets):
    """Calculate the ngrams of the word and hash them.

    Parameters
    ----------
    word : str
        The word to calculate ngram hashes for.
    minn : int
        Minimum ngram length
    maxn : int
        Maximum ngram length
    num_buckets : int
        The number of buckets

    Returns
    -------
        A list of hashes (integers), one per each detected ngram.

    """
    encoded_ngrams = compute_ngrams_bytes(word, minn, maxn)
    hashes = [ft_hash_bytes_my(n) % num_buckets for n in encoded_ngrams]
    return hashes


class FastTextWrapper:

    def __init__(self, vec_path, vocab_path, vectors_ngram_path, min_n, max_n, bucket, count=0, dtype=torch.float32):
        self.vec_path = vec_path
        self.vocab_path = vocab_path
        self.vectors_ngram_path = vectors_ngram_path
        self.vectors = None
        self.vectors_ngrams = None
        self.key_to_index = None
        self.min_n = min_n
        self.max_n = max_n
        self.bucket = bucket
        self.vector_size = config.embedding_size
        self.no_of_words = None
        self.vectors_vocab = torch.zeros((count, self.vector_size), dtype=dtype, device=config.device)
        self.expandos = {}
        self.load_vectors()
        self.oov_idx = len(self.key_to_index)
        self.oov_idx_to_word = {}
        self.oov_word_to_idx = {}


    def load_vectors(self):
        self.vectors = torch.load(self.vec_path).to(config.device)
        self.vectors_ngrams = torch.load(self.vectors_ngram_path).to(config.device)
        with open(self.vocab_path, 'r') as f:
            self.key_to_index = json.load(f)
        self.index_to_key = list(self.key_to_index.keys())

        # freeze the embedding layers
        self.vectors_vocab.requires_grad = False
        self.vectors.requires_grad = False
        self.vectors_ngrams.requires_grad = False

    def get_vector(self, word, norm=False):
            """Get word representations in vector space, as a 1D numpy array.
            """
            if word in self.key_to_index:
                return self.vectors[self.key_to_index[word]]
            elif self.bucket == 0:
                raise KeyError('cannot calculate vector for OOV word without ngrams')
            elif word == config.UNK_TOKEN:
                return torch.zeros((self.vectors_ngrams.shape[1]), dtype=torch.float32, device=config.device)
            else:
                word_vec = torch.zeros((self.vectors_ngrams.shape[1]), dtype=torch.float32, device=config.device)
                ngram_weights = self.vectors_ngrams
                ngram_hashes = ft_ngram_hashes(word, self.min_n, self.max_n, self.bucket)
                if len(ngram_hashes) == 0:
                    print('could not extract any ngrams from %r, returning origin vector', word)
                    return word_vec
                for nh in ngram_hashes:
                    word_vec += ngram_weights[nh]
                if norm:
                    return word_vec / torch.norm(word_vec)
                else:
                    return word_vec / len(ngram_hashes)
    
    def get_index(self, word):
        '''
        give a word, return its index in the embedding matrix.
        If the word is not in the embedding matrix, adds it a dict and return
        a new index.
        '''
        if word in self.key_to_index:
            return self.key_to_index[word]
        else:
            if word == config.UNK_TOKEN:
                return -100
            if word in self.oov_word_to_idx:
                return self.oov_word_to_idx[word]
            self.oov_word_to_idx[word] = self.oov_idx
            self.oov_idx_to_word[self.oov_idx] = word
            self.oov_idx += 1
            return self.oov_idx - 1
    
    def __call__(self, *args, **kwds):

        if len(args) == 1:
            batch = args[0]
        

        output = torch.zeros((len(batch), len(batch[0]), config.embedding_size), dtype=torch.float32, device=config.device)
        for i, sequence in enumerate(batch):
            for j, idx in enumerate(sequence):
                if idx < len(self.key_to_index):
                    output[i, j, :] = self.vectors[idx]
                elif int(idx) in self.oov_idx_to_word:
                    output[i, j, :] = self.get_vector(self.oov_idx_to_word[int(idx)])
                else:
                    print(self.oov_idx_to_word)
                    output[i, j, :] = torch.zeros((config.embedding_size), dtype=torch.float32, device=config.device)
        
        return output

                    
                

# if __name__ == '__main__':
    
#     test_words = ['hello', 'dark_souls', 'new_york', 'dark_souls_3', 'Michael_Jackson']

#     print('loading model')

#     model = FastText.load('model/fasttext/fasttext-wiki-news-subwords-300.bin')

#     print('model loaded')

#     # subtract and save the vocab

#     vectors_path = 'model/fasttext/fasttext-wiki-news-subwords-300-vectors.pt'
#     vocab_path = 'model/fasttext/fasttext-wiki-news-subwords-300-vocab.json'
#     vectors_ngrams_path = 'model/fasttext/fasttext-wiki-news-subwords-300-vectors_ngrams.pt'

#     vocab = model.wv.key_to_index
#     vectors = model.wv.vectors
#     weights = torch.FloatTensor(vectors)
#     torch.save(weights, vectors_path)
#     torch.save(torch.from_numpy(model.wv.vectors_ngrams), vectors_ngrams_path)

#     with open(vocab_path, 'w') as f:
#         json.dump(vocab, f)

#     # Now load my wrapper and check if it outputs the same vectors

#     ft = FastTextWrapper(vectors_path, vocab_path, vectors_ngrams_path, 3, 6, 2000000)

    
#     vectors = []
#     for word in test_words:
#         vectors.append(ft.get_vector(word).numpy())

#     vectors2 = []
#     for word in test_words:
#         vectors2.append(model.wv[word])
    
#     for i in range(len(test_words)):
#         print(np.allclose(vectors[i], vectors2[i]))
    