from abc import ABC, abstractmethod

class Embedding(ABC):

    @abstractmethod
    def get_embeddings(self):
        pass