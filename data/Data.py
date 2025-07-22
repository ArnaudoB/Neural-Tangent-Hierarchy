import jax.numpy as jnp
from jax import random

class Data:

    def __init__(self, d:int, n:int, key=None):
        self.d = d
        self.n = n
        
        if key is None:
            key = random.PRNGKey(0)

        self.x, self.y = self.init_data(key)


    def init_data(self, key):
        """
        Initializes the data such that Assumption 2.2 is verified. Assumption 2.2 states that the data (x_i) has to verify :
        - There exists c > 0 s.t. c < x_i <= c^-1 for all x_i (to enforce that it suffices to normalize the x_i)
        - For all r <= d, for all 1 <= i_1 < ... < i_r <= n, the matrix [x_i1, ..., x_ir] has all its singular values > 0
        To enforce the second condition, it suffices that every sub-family of size d of (x_i) is free, which has probability 1 to occur if, for example,
        the x_i are i.i.d and follow a law with a density (theorem).
        
        """
        keys = random.split(key, 2)
        key_x, key_y = keys
        
        x = random.normal(key_x, (self.n, self.d, 1))
        x = x / jnp.linalg.norm(x, axis=1, keepdims=True)
        
        y = random.uniform(key_y, (self.n, 1, 1), minval=-1.0, maxval=1.0)
        
        return x, y
    

    def __len__(self):
        return self.n
    
    
if __name__ == '__main__':
    data = Data(3, 8)
    print(data.x)