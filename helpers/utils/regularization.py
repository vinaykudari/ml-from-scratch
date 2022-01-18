import numpy as np

class LassoRegularization:
    'L1 Regularization'
    
    def __init__(self, alpha):
        self.alpha = alpha
    
    def __call__(self, w):
        return self.alpha * np.sum(w)
    
    def grad(self, w):
        return self.alpha * np.sign(w)
    
    
class RidgeRegularization:
    'L2 Regularization'
    
    def __init__(self, alpha):
        self.alpha = alpha
        
    def __call__(self, w):
        return self.alpha * 0.5 * w.dot(w.T)
    
    def grad(self, w):
        return self.alpha * w
    
    
class ElasticNetRegularization:
    'L1 and L2 Regularization'
    
    def __init__(self, alpha, l1_ratio=0.5):
        assert l1_ratio <= 1.0
        assert alpha != 0
        
        self.alpha = alpha
        self.l1_reg = LassoRegularization(l1_ratio)
        self.l2_reg = RidgeRegularization(1 - l1_ratio)
        
    def __call__(self, w):
        return (self.l1_reg(w) + self.l2_reg(w))
    
    def grad(self, w):
        return (self.l1_reg.grad(w) + self.l2_reg.grad(w))
        