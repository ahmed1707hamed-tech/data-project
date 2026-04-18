from abc import ABC, abstractmethod

class BaseModelConfig(ABC):
    @abstractmethod
    def load_model(self):
        pass
        
    @abstractmethod
    def predict(self, text: str):
        pass
