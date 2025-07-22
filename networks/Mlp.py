import jax
import jax.numpy as jnp
from jax import random, jit, vmap


class Mlp:


    def __init__(self, d: int, m: int, h: int, sigma: str, sigma_w: float, sigma_a: float, key=None):
        self.d = d
        self.m = m
        self.h = h
        self.sigma_w = sigma_w
        self.sigma_a = sigma_a
        
        if key is None:
            key = random.PRNGKey(0)
            
        self.params = self._init_params(key)

        if sigma.lower() == "relu":
            self.activation = jax.nn.relu
            self.activation_prime = lambda x: (x > 0).astype(jnp.float32)
        elif sigma.lower() == "tanh":
            self.activation = jnp.tanh
            self.activation_prime = lambda x: 1 - jnp.tanh(x)**2
        else :
            raise ValueError(f"Unknown activation: {sigma}")
        
        # JIT compile for speed
        self.forward_with_params_jit = jit(self._forward_with_params)
        self.forward_with_outputs_jit = jit(self._forward_with_outputs)
        self.forward_batch_with_outputs_jit = jit(vmap(self._forward_with_outputs, in_axes=(None, 0)))
        self.ntk_jit = jit(self._ntk)
        self.ntk_autodiff_jit = jit(self._ntk_autodiff)

        
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
    

    def _forward_with_params(self, params, x):
        z = x
        for W in params[:-1]:
            z = self.activation(W @ z) / jnp.sqrt(self.m)
        z = params[-1].T @ z
        return z
    
    def forward_with_params(self, params, x):
        return self.forward_with_params_jit(params, x)
    
    def forward(self, x):
        return self.forward_with_params_jit(self.params, x)
    


    def _forward_with_outputs(self, x):
        z = x
        intermediates = jnp.zeros((self.h, self.m, 1))
        for i in range(self.h):
            z = self.activation(self.params[i] @ z) / jnp.sqrt(self.m)
            intermediates = intermediates.at[i].set(z)
        
        z = self.params[-1].T @ z

        return z, intermediates
        
    
    def forward_with_outputs(self, x, batch=False):
        if batch:
            return self.forward_batch_with_outputs_jit(x)
        return self.forward_with_outputs_jit(x)
    

    

    def _ntk(self, x_l, x_r):

        _, intermediates_l = self.forward_with_outputs(x_l)
        _, intermediates_r = self.forward_with_outputs(x_r)

        ntk = intermediates_l[-1].T @ intermediates_r[-1] # G^(H + 1)

        # initialize prefixes
        last_hidden_l = intermediates_l[-2]
        last_hidden_r = intermediates_r[-2]
        
        prefix_l = jnp.diag(self.activation_prime(self.params[-2] @ last_hidden_l).flatten()) @ self.params[-1] / jnp.sqrt(self.m) # .flatten() because @ returns a (n, 1)-shaped element here
        prefix_r = jnp.diag(self.activation_prime(self.params[-2] @ last_hidden_r).flatten()) @ self.params[-1] / jnp.sqrt(self.m)
        
        for layer in range(self.h - 1, -1, -1):

            if layer > 0:
                ntk += (prefix_l.T @ prefix_r) * (intermediates_l[layer-1].T @ intermediates_r[layer-1])
                # update prefixes 
                if layer == 1:
                    prev_layer_l = x_l
                    prev_layer_r = x_r
                else:
                    prev_layer_l = intermediates_l[layer-2]  # Fix: layer-2 not layer
                    prev_layer_r = intermediates_r[layer-2]
                
                prefix_l = jnp.diag(self.activation_prime(self.params[layer-1] @ prev_layer_l).flatten()) @ self.params[layer].T @ prefix_l / jnp.sqrt(self.m)
                prefix_r = jnp.diag(self.activation_prime(self.params[layer-1] @ prev_layer_r).flatten()) @ self.params[layer].T @ prefix_r / jnp.sqrt(self.m)
            else:
                ntk += (prefix_l.T @ prefix_r) * (x_l.T @ x_r) # input layer
            
        return ntk.squeeze()

    def ntk(self, x_l, x_r):
        return self.ntk_jit(x_l, x_r)
    

    

    def _ntk_autodiff(self, x_l, x_r):

        def network_fn(params, x):
            z = x
            for i in range(self.h):
                z = self.activation(params[i] @ z) / jnp.sqrt(self.m)
            z = params[-1].T @ z
            return z.squeeze()
        
        jac_fn = jax.jacfwd(network_fn, argnums=0)
        jac_l = jac_fn(self.params, x_l)
        jac_r = jac_fn(self.params, x_r)
        
        jac_l_flat = jnp.concatenate([jnp.ravel(layer_grad) for layer_grad in jac_l])
        jac_r_flat = jnp.concatenate([jnp.ravel(layer_grad) for layer_grad in jac_r])
        
        return jnp.dot(jac_l_flat, jac_r_flat)
    
    def ntk_autodiff(self, x_l, x_r):
        return self.ntk_autodiff_jit(x_l, x_r)



if __name__ == "__main__":
    mlp = Mlp(3, 5, 5, "relu", 1.0, 1.0)
    x = jnp.array([[2.0], [3.0], [4.0]])
    y = jnp.array([[-1.0], [5.0], [2.0]])
    ntk = mlp.ntk(x, y)
    print(ntk)
    ntk_diff = mlp.ntk_autodiff(x, y)
    print(ntk_diff)
    