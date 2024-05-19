from abc import ABC, abstractmethod


class Attacker(ABC):
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @abstractmethod
    def generate(self, sounds, targets):
        raise NotImplementedError
