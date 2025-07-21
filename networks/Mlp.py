import jax
import jax.numpy as jnp
from jax import random, jit, vmap

class MLP:
    def __init__(self, d: int, m: int, h: int, sigma: str, sigma_w: float, sigma_a: float, key=None):
        self.d = d
        self.m = m
        self.h = h
        self.sigma_w = sigma_w
        self.sigma_a = sigma_a
        
        if key is None:
            key = random.PRNGKey(0)
            
        self.activation = self._get_activation(sigma)
        self.params = self._init_params(key)
        
        # JIT compile for speed
        self.forward_jit = jit(self._forward)
        self.forward_batch_jit = jit(vmap(self._forward, in_axes=(None, 0)))

        
    def _get_activation(self, sigma):
        if sigma.lower() == "relu":
            return jax.nn.relu
        elif sigma.lower() == "tanh":
            return jnp.tanh
        else:
            raise ValueError(f"Unknown activation: {sigma}")
        
    
    def _init_params(self, key):
        keys = random.split(key, self.h + 1)
        params = []
        
        # input layer
        w1 = random.normal(keys[0], (self.m, self.d)) * self.sigma_w
        params.append(w1)
        
        # hidden layers
        for i in range(1, self.h):
            w = random.normal(keys[i], (self.m, self.m)) * self.sigma_w
            params.append(w)
        
        # output coefficients
        a = random.normal(keys[-1], (self.m, 1)) * self.sigma_a
        params.append(a)
        
        return params
    
    def _forward(self, x):
        z = x
        intermediates = [z]
        
        for i in range(self.h):
            z = self.activation(self.params[i] @ z) / jnp.sqrt(self.m)
            intermediates.append(z)
        
        z = self.params[-1].T @ z
        
        intermediates.append(z)
        
        return intermediates
    
    def forward(self, x, batch=False):
        if batch:
            return self.forward_batch_jit(x)
        return self.forward_jit(x)
    
if __name__ == "__main__":
    mlp = MLP(3, 5, 5, "relu", 1.0, 1.0)
    x = jnp.array([[2.0], [3.0], [4.0]])
    output = mlp.forward(x)
    print(output)
    