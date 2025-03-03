from scipy.sparse import coo_matrix, save_npz

__all__ = [
        'SpinHalfOperator',
        ]


class SpinHalfOperator:
    
    def __init__(self, disk=False, path=None):
        
        if disk == True:
            self.load_disk(path)
            
    @property
    def sx(self):
        return coo_matrix(([0.5, 0.5], ([0, 1], [1, 0])), shape=(2, 2))
    
    @property
    def sy(self):
        return coo_matrix(([-1j/2, 1j/2], ([0, 1], [1, 0])), shape=(2, 2))
    
    @property
    def sz(self):
        return coo_matrix(([0.5, -0.5], ([0, 1], [0, 1])), shape=(2, 2))
    
    @property
    def sp(self):
        return coo_matrix(([1.], ([0], [1])), shape=(2, 2))
    
    @property
    def sm(self):
        return coo_matrix(([1.], ([1], [0])), shape=(2, 2))
    
    @property
    def identity(self):
        return coo_matrix(([1., 1.], ([0, 1], [0, 1])), shape=(2, 2))
        
    def load_disk(self, path):
        
        if path is None:
            raise ValueError('Here need a path to save operator.')
            
        save_npz(path+'/sx', self.sx)
        save_npz(path+'/sy', self.sy)        
        save_npz(path+'/sz', self.sz)        
        save_npz(path+'/sp', self.sp)        
        save_npz(path+'/sm', self.sm)