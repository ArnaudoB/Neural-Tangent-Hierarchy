import jax
import jax.numpy as jnp
from jax.tree_util import tree_leaves
from typing import List

class NTH:

    def __init__(self, model):
        self.model = model
        self.params = model.params

        self._val_fn = model.forward_with_params

        self._val_and_grad_fn = jax.value_and_grad(self._val_fn)

    @staticmethod
    def _flatten(tree):
        return jnp.concatenate([jnp.ravel(x) for x in tree_leaves(tree)])
    
    def __call__(self, x_list: List[jnp.ndarray], r: int):
        assert r >= 2, "Kernel order r must be >= 2"
        assert len(x_list) == r, "Provide r input vectors"

        return self._Kr(self.params, x_list, r)

    def _Kr(self, params, x_list: List[jnp.ndarray], r: int) -> jnp.ndarray:
        if r == 2:
            return self._K2(params, x_list[0], x_list[1])

        def K_prev_fn(p):
            return self._Kr(p, x_list[:-1], r - 1)

        grad_K_prev = jax.grad(K_prev_fn)(params)

        _, grad_fx = self._val_and_grad_fn(params, x_list[-1])

        return jnp.dot(self._flatten(grad_fx), self._flatten(grad_K_prev))

    def _K2(self, params, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        _, grad1 = self._val_and_grad_fn(params, x1)
        _, grad2 = self._val_and_grad_fn(params, x2)
        return jnp.dot(self._flatten(grad1), self._flatten(grad2))

if __name__ == "__main__":
    from networks.Mlp import Mlp
    mlp = Mlp(3, 5, 5, "relu", 1.0, 1.0)
    hierarchy = NTH(mlp)
    x = jnp.array([[2.0], [35.0], [4.0]])
    y = jnp.array([[-1.0], [45.0], [2.0]])
    z = jnp.array([[-1.0], [8.0], [-40.0]])
    t = jnp.array([[-7.0], [1.0], [-320.0]])
    print(hierarchy([x, y, z], 3))
