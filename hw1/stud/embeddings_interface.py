from abc import ABC, abstractmethod

class Embedding(ABC):
    '''
    Simple interface for embeddings
    '''
    @abstractmethod
    def get_embeddings(self):
        pass