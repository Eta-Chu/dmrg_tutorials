from copy import deepcopy
import numpy as np

from ..model import Block


__all__ = [
    'finiteDMRG',

]


class finiteDMRG:

    def __init__(self, block: Block, length: int, m: int):
        if length % 2 != 0:
            raise InterruptedError('Only supported even length now.')
        self.length = length
        self.halfLength = length // 2
        self.wave_func = np.zeros(shape=(length), dtype=object)
        self.block_list = np.zeros(shape=(length), dtype=Block)
        self.log = {
            'energy': [],
            'spectrum': [],
            'truncation_error': []
        }
        self.sys_block = block
        self.env_block = block
        self.direction = None
        self.center = 0

    def warmup(self, m: int):
        self.direction = 'warmup'
        self.block_list[0] = deepcopy(self.sys_block)
        self.block_list[-1] = deepcopy(self.sys_block)
        for i in range(1, self.halfLength):
            self.center = i
            self.sys_block.enlarge()
            self.block_list[i] = deepcopy(self.sys_block)
            self.block_list[-(i+1)] = deepcopy(self.sys_block)
            energy, spectrum, projector, truncation_err = self.sys_block.renormalization(m)
            self.wave_func[i] = self.wave_func[-(i+1)] = projector
            self.log['energy'].append(energy)
            self.log['spectrum'].append(spectrum)
            self.log['truncation_error'].append(truncation_err)
            self.__visualization()


    def sweep(self, m: int):
        self.direction = 'LtoR'
        for i in range(self.halfLength, self.length - 2):
            self.center = i
            self.sys_block.enlarge()
            self.env_block = self.block_list[i+1]
            self.block_list[i] = deepcopy(self.sys_block)
            superB = self.sys_block.fuse(self.env_block)
            energy, spectrum, projector, truncation_err = superB.renormalization(m)
            self.wave_func[i] = projector
            self.log['energy'].append(energy)
            self.log['spectrum'].append(spectrum)
            self.log['truncation_error'].append(truncation_err)
            self.__visualization()

        self.sys_block = deepcopy(self.block_list[-1])
        self.direction = 'RtoL'
        for i in range(self.length-2, 1, -1):
            self.center = i
            self.sys_block.enlarge()
            self.env_block = self.block_list[i-1]
            self.block_list[i] = deepcopy(self.sys_block)
            superB = self.sys_block.fuse(self.env_block)
            energy, spectrum, projector, truncation_err = superB.renormalization(m)
            self.wave_func[i] = projector
            self.log['energy'].append(energy)
            self.log['spectrum'].append(spectrum)
            self.log['truncation_error'].append(truncation_err)
            self.__visualization()

        self.sys_block = deepcopy(self.block_list[0])
        self.direction = 'LtoR'
        for i in range(1, self.halfLength):
            self.center = i
            self.sys_block.enlarge()
            self.env_block = self.block_list[i+1]
            self.block_list[i] = deepcopy(self.sys_block)
            superB = self.sys_block.fuse(self.env_block)
            energy, spectrum, projector, truncation_err = superB.renormalization(m)
            self.wave_func[i] = projector
            self.log['energy'].append(energy)
            self.log['spectrum'].append(spectrum)
            self.log['truncation_error'].append(truncation_err)
            self.__visualization()

    def run(self, sweep_number: int, m: int):
        self.warmup(m)
        for i in range(sweep_number):
            print(f'This is {i}th sweep process:')
            self.sweep(m)

    def __visualization(self):
        if self.direction == 'LtoR':
            print('='*self.center + '**' + '=' * (self.length - self.center - 2))
            print('energy:', self.log['energy'][-1], 'truncation error:', self.log['truncation_error'][-1])
        elif self.direction == 'RtoL':
            print('='*(self.center-1) + '**' + '=' * (self.length - self.center - 1))
            print('energy:', self.log['energy'][-1], 'truncation error:', self.log['truncation_error'][-1])
        elif self.direction == 'warmup':
            print('='*self.center + '**' + '='*self.center)
            print('energy:', self.log['energy'][-1], 'truncation error:', self.log['truncation_error'][-1])