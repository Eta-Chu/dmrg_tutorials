import numpy as np
from copy import deepcopy
from ..model import Block

__all__ = [
        'infiniteDMRG',

        ]

class infiniteDMRG:

    def __init__(self, B: Block):
        self.block = B
        self.cache = [deepcopy(B)]
        self.log = {
            'energy': [],
            'entropy': [],
            'spectrum': [],
            'error': [],
            'dimension': []
        }
    
    def entanglement_entropy(self, spectrum):
        spectrum = spectrum[np.where(spectrum > 1e-16)[0]]
        return -spectrum @ np.log(spectrum)

    def run(self, iteration: int, m: int, save=False):
        for i in range(iteration):
            self.block.enlarge()
            energy, spectrum, _, truncation_err = self.block.renormalization(m)
            self.log['energy'].append(energy)
            self.log['spectrum'].append(spectrum)
            self.log['entropy'].append(self.entanglement_entropy(spectrum))
            self.log['error'].append(truncation_err)
            self.log['dimension'].append(self.block.dimension)
            if save:
                self.cache.append(deepcopy(self.block))