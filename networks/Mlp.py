import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from jax.tree_util import tree_leaves


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
            self.activation_prime = lambda x: jnp.where(x > 0, 1.0, 0.0)
        elif sigma.lower() == "tanh":
            self.activation = jnp.tanh
            self.activation_prime = lambda x: 1 - jnp.tanh(x)**2
        else :
            raise ValueError(f"Unknown activation: {sigma}")

        self.val_and_grad_fn = jax.jit(jax.value_and_grad(self.forward_with_params))


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
    
    @staticmethod
    @partial(jax.jit, static_argnames=["activation", "m"])
    def _forward_with_params(params, x, activation, m):
        z = x
        for W in params[:-1]:
            z = activation(W @ z) / jnp.sqrt(m)
        z = params[-1].T @ z
        return z.squeeze()
    
    def forward_with_params(self, params, x):
        return Mlp._forward_with_params(params, x, self.activation, self.m)
    
    def forward(self, x):
        return Mlp._forward_with_params(self.params, x)
    

    @staticmethod
    @partial(jax.jit, static_argnames=["activation", "h", "m"])
    def _forward_with_outputs(params, x, activation, h, m):
        z = x
        intermediates = jnp.zeros((h, m, 1))
        for i in range(h):
            z = activation(params[i] @ z) / jnp.sqrt(m)
            intermediates = intermediates.at[i].set(z)
        z = params[-1].T @ z
        return z, intermediates
        
    
    def forward_with_outputs(self, x):
        return Mlp._forward_with_outputs(self.params, x, self.activation, self.h, self.m)
    

    @staticmethod
    @partial(jax.jit, static_argnames=["activation_prime", "h", "m"])
    def _ntk(x_l, x_r, intermediate_outputs_l, intermediate_outputs_r, params, activation_prime, h, m):

        sqrt_m = jnp.sqrt(m)
        ntk = intermediate_outputs_l[-1].T @ intermediate_outputs_r[-1] # G^(H + 1)

        # initialize prefixes
        
        prefix_l = jnp.diag(activation_prime(params[-2] @ intermediate_outputs_l[-2]).flatten()) @ params[-1] / sqrt_m  # .flatten() because @ returns a (n, 1)-shaped element here
        prefix_r = jnp.diag(activation_prime(params[-2] @ intermediate_outputs_r[-2]).flatten()) @ params[-1] / sqrt_m 
        
        for layer in range(h - 1, -1, -1):
            if layer > 0:
                ntk += (prefix_l.T @ prefix_r) * (intermediate_outputs_l[layer-1].T @ intermediate_outputs_r[layer-1])
                # update prefixes 
                if layer == 1:
                    prev_layer_l = x_l
                    prev_layer_r = x_r
                else:
                    prev_layer_l = intermediate_outputs_l[layer-2]
                    prev_layer_r = intermediate_outputs_r[layer-2]
                
                prefix_l = jnp.diag(activation_prime(params[layer-1] @ prev_layer_l).flatten()) @ params[layer].T @ prefix_l / sqrt_m 
                prefix_r = jnp.diag(activation_prime(params[layer-1] @ prev_layer_r).flatten()) @ params[layer].T @ prefix_r / sqrt_m 
            else:
                ntk += (prefix_l.T @ prefix_r) * (x_l.T @ x_r) # input layer
        return ntk.squeeze()

    def ntk(self, x_l, x_r):
        _, intermediate_outputs_l = self.forward_with_outputs(x_l)
        _, intermediate_outputs_r = self.forward_with_outputs(x_r)
        return Mlp._ntk(x_l, x_r, intermediate_outputs_l, intermediate_outputs_r, self.params, self.activation_prime, self.h, self.m)
    
    @staticmethod
    @partial(jax.jit, static_argnames=["activation", "h", "m"])
    def _ntk_autodiff(params, x_l, x_r, activation, h, m):
        def network_fn(p, x):
            z = x
            for i in range(h):
                z = activation(p[i] @ z) / jnp.sqrt(m)
            z = p[-1].T @ z
            return z.squeeze()
        jac_fn = jax.jacfwd(network_fn, argnums=0)
        jac_l = jac_fn(params, x_l)
        jac_r = jac_fn(params, x_r)
        jac_l_flat = jnp.concatenate([jnp.ravel(g) for g in jac_l])
        jac_r_flat = jnp.concatenate([jnp.ravel(g) for g in jac_r])
        return jnp.dot(jac_l_flat, jac_r_flat)

    
    def ntk_autodiff(self, x_l, x_r):
        return Mlp._ntk_autodiff(self.params, x_l, x_r, self.activation, self.h, self.m)
    

    def Kr(self, params, x, r: int):
        assert r >= 2, "Kernel's order must be greater than or equal to 2"
        assert len(x) == r, "Number of inputs must match kernel order"

        if r == 2:
            return self._ntk_autodiff(params, x[0], x[1], self.activation, self.h, self.m)
        else:
            def K_prev(p):
                return self.Kr(p, x[:-1], r - 1)
            dK_prev = jax.grad(K_prev)(params)

            _, grads = self.val_and_grad_fn(params, x[-1])
            return jnp.dot(Mlp.flatten_grads(grads), Mlp.flatten_grads(dK_prev))

    @staticmethod
    def flatten_grads(grad_tree):
        return jnp.concatenate([jnp.ravel(g) for g in tree_leaves(grad_tree)])



if __name__ == "__main__":
    mlp = Mlp(3, 5, 5, "relu", 1.0, 1.0)
    x = jnp.array([[2.0], [35.0], [4.0]])
    y = jnp.array([[-1.0], [45.0], [2.0]])
    z = jnp.array([[-1.0], [8.0], [-40.0]])
    t = jnp.array([[-7.0], [1.0], [-320.0]])
    arg = [x, y, z]
    ntk3 = mlp.Kr(mlp.params, arg, 3)
    print(ntk3)
    