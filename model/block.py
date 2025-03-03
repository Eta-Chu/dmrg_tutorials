from scipy.sparse import kron, identity
from scipy.sparse.linalg import LinearOperator, eigsh
import numpy as np

from .operator import SpinHalfOperator

sz = SpinHalfOperator().sz
sp = SpinHalfOperator().sp
sm = SpinHalfOperator().sm


__all__ = [
        'Block',
        'SuperBlock',
        'SpinHalfXXZChain',
        'SpinHalfXXZChainSuperBlock',

        ]


class Block(object):

    def __init__(self):
        self.length = 1
        self.d = 0
        self.dimension = 0

    def initial(self):
        raise NotImplementedError('This should be impleted in subclass.')

    def enlarge(self):
        raise NotImplementedError('This should be impleted in subclass.')
    
    def fuse(self, b: "Block") -> "SuperBlock":
        raise NotImplementedError('This should be impleted in subclass.')


class SuperBlock(object):

    def __init__(self, lblock: Block, rblock: Block):
        self.lBlock = lblock
        self.rBlock = rblock

    def fuse_ham(self):

        raise NotImplementedError('This should be impleted in subclass.')

    def matvec(self, v0=None):

        raise NotImplementedError('This should be impleted in subclass.')

    def eigen(self, v0=None):

        raise NotImplementedError('This should be impleted in subclass.')

    def renormalization(self, m: int, vo=None):

        raise NotImplementedError('This should be impleted in subclass.')


class SpinHalfXXZChain(Block):

    def __init__(self, delta: float, hz: float):
        super().__init__()
        self.delta = delta
        self.hz = hz
        self.d = 2
        self.length = 0
        self.dimension = 0
        self.ham = {
                'completed': None,
                'connected': None,
                'PLPR': None, # for superblock matvec()
                }
        self.initial()

    def initial(self):
        '''generate one-site Block hamitonian.
        
        Model-dependent
        '''
        # one-site interaction
        self.ham['completed'] = self.hz * sz

        # two-site interaction
        self.ham['connected'] = {
            'S+S-': (1/2) * sp,
            'S-S+': (1/2) * sm,
            'SzSz': self.delta * sz,
        }

        # superblock PLPR interaction
        self.ham['PLPR'] = (
            self.delta * kron(sz, sz) + 
            (1 / 2) * kron(sp, sm) + 
            (1 / 2) * kron(sm, sp)
        )

        self.length = 1
        self.dimension = self.d

    def enlarge(self):
        '''enlarge the Block and construct the Hamitonian of enlarged Block.
        Model-dependent
        '''
        self.ham['completed'] = kron(self.ham['completed'], identity(self.d))

        self.ham['completed'] += self.hz * kron(identity(self.dimension), sz)
        self.ham['completed'] += (
            kron(self.ham['connected']['S+S-'], sm) +
            kron(self.ham['connected']['S-S+'], sp) +
            kron(self.ham['connected']['SzSz'], sz)
        )

        self.ham['connected'] = {
            'S+S-': (1/2) * kron(identity(self.dimension), sp),
            'S-S+': (1/2) * kron(identity(self.dimension), sm),
            'SzSz': self.delta * kron(identity(self.dimension), sz),
        }
        self.dimension = self.dimension * self.d
        self.length += 1
    
    def matvec(self, v0=None):
        '''
        '''
        d = self.d
        m = self.dimension

        v0 = v0.reshape(m, m)
        v = v0.reshape(m // d, d, m // d, d).transpose([1, 3, 0, 2]).reshape([d * d, -1])
        res = (
            self.ham['completed'].dot(v0) + 
            (self.ham['completed'].dot(v0.T)).T + 
            self.ham['PLPR'].dot(v).reshape([d, d, m // d, m // d]).transpose([2, 0, 3, 1]).reshape([m, m])
        )
        
        return res.reshape([-1])

    def eigen(self, v0=None):
        D = self.dimension * self.dimension
        A = LinearOperator((D, D), matvec=self.matvec)
        energy, vector = eigsh(A, which='SA', v0=v0, k=1)
        
        return energy.item(), vector.reshape(self.dimension, self.dimension)
    
    def benchmark(self):
        ham = kron(self.ham['completed'], identity(self.dimension)) + kron(identity(self.dimension), self.ham['completed'])
        ham += (
            kron(self.ham['connected']['S+S-'], 2 * self.ham['connected']['S-S+']) +
            kron(self.ham['connected']['S-S+'], 2 * self.ham['connected']['S+S-'])
        )
        if self.delta != 0:
            ham += kron(self.ham['connected']['SzSz'], (1 / self.delta) * self.ham['connected']['SzSz'])
            
        return ham
    
    def renormalization(self, m: int, v0=None):
        energy, psi = self.eigen(v0)
        rho = np.dot(psi, psi.T.conj())
        spectrum, vec = np.linalg.eigh(rho)
        vec = vec[:, ::-1]
        dim = min(m, self.dimension)
        truncation_err = np.sum(spectrum[:-dim])
        spectrum = spectrum[:-(dim+1):-1]
        vec = vec[:, :dim]
        self.ham['completed'] = vec.T.conj().dot(self.ham['completed'].dot(vec))
        for key in self.ham['connected'].keys():
            self.ham['connected'][key] = vec.T.conj().dot(self.ham['connected'][key].dot(vec))
        self.dimension = dim

        return energy, spectrum, vec, truncation_err
    
    def fuse(self, b: "SpinHalfXXZChain") -> "SpinHalfXXZChainSuperBlock":

        return SpinHalfXXZChainSuperBlock(self, b)


    def __repr__(self):

        return f"block dimension is {self.dimension}, "\
            "block length is {self.length}"


class SpinHalfXXZChainSuperBlock(SuperBlock):

    def __init__(self, lblock: SpinHalfXXZChain, rblock: SpinHalfXXZChain):
        super().__init__(lblock, rblock)
        self.lDim = lblock.dimension
        self.rDim = rblock.dimension
        self.delta = rblock.delta
        self.hz = rblock.hz
        self.d = rblock.d

        self.ham = {
            'BLPL': lblock.ham['completed'],
            'PRBR': rblock.ham['completed'],
            'PLPR': (
                self.delta * kron(sz, sz) + 
                (1 / 2) * kron(sp, sm) + 
                (1 / 2) * kron(sm, sp)
            )
        }

    def matvec(self, v0=None):
        '''
        '''
        d = self.d

        v0 = v0.reshape(self.lDim, self.rDim)
        v = v0.reshape(self.lDim // d, d, self.rDim // d, d).transpose([1, 3, 0, 2]).reshape([d * d, -1])
        res = (
            self.ham['BLPL'].dot(v0) + 
            (self.ham['PRBR'].dot(v0.T)).T + 
            self.ham['PLPR'].dot(v).reshape([d, d, self.lDim // d, self.rDim // d]).transpose([2, 0, 3, 1]).reshape([self.lDim, self.rDim])
        )
        
        return res.reshape([-1])

    def eigen(self, v0=None):
        m = self.lDim * self.rDim
        A = LinearOperator((m, m), matvec=self.matvec)
        energy, vector = eigsh(A, which='SA', v0=v0, k=1)
        
        return energy.item(), vector.reshape(self.lDim, self.rDim)

    def renormalization(self, m: int, v0=None):
        energy, psi = self.eigen(v0)
        rho = np.dot(psi, psi.T.conj())
        spectrum, vec = np.linalg.eigh(rho)
        vec = vec[:, ::-1]
        dim = min(m, self.lDim)
        truncation_err = np.sum(spectrum[:-dim])
        spectrum = spectrum[:-(dim+1):-1]
        vec = vec[:, :dim]
        self.lBlock.ham['completed'] = vec.T.conj().dot(self.lBlock.ham['completed'].dot(vec))
        for key in self.lBlock.ham['connected'].keys():
            self.lBlock.ham['connected'][key] = vec.T.conj().dot(self.lBlock.ham['connected'][key].dot(vec))
        self.lBlock.dimension = dim

        return energy, spectrum, vec, truncation_err