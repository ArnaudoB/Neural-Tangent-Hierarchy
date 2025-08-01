from functools import partial
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_leaves

class GradientFlow:

    def __init__(self, model, data, loss_fn, lr):
        self.model = model
        self.data = data
        self.x = data.x
        self.y = data.y
        self.lr = lr
        self.history = {"loss": [], "grad_norm": []}

        if loss_fn.lower() == "mse": 
            self.loss_fn = lambda x, y : jnp.mean(jnp.pow(x - y, 2))/2
            
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")
        
        # create loss-and-grad function just once
        def loss_on_params(params):
            preds = self.model.forward_with_params(params, self.x)
            return self.loss_fn(preds, self.y)

        self.loss_and_grad_fn = jax.jit(jax.value_and_grad(loss_on_params))
        
    
    
    @staticmethod
    @partial(jax.jit, static_argnames=["dt", "loss_and_grad_fn"])
    def _compute_gd_step(params, dt, loss_and_grad_fn):
        val, grads = loss_and_grad_fn(params)
        new_params = tree_map(lambda p, g: p - dt * g, params, grads)
        grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in tree_leaves(grads)))
        return new_params, val, grad_norm
    
    
    def step(self, params, dt=None):
        if dt is None:
            dt = self.lr
        new_params, loss_value, grad_norm = GradientFlow._compute_gd_step(params, dt, self.loss_and_grad_fn)
        self.history["loss"].append(float(loss_value))
        self.history["grad_norm"].append(float(grad_norm))
        return new_params
    
    
    def train(self, params, num_steps, dt=None, verbose=True):
        current_params = params
        for step in range(num_steps):
            current_params = self.step(current_params, dt)
            if verbose and (step + 1) % 100 == 0:
                current_loss = self.history["loss"][-1]
                current_grad_norm = self.history["grad_norm"][-1]
                print(f"Step {step + 1:4d}: Loss = {current_loss:.6f}, "
                      f"Grad Norm = {current_grad_norm:.6f}")
        return current_params
    

    def get_current_loss(self):
        return self.history["loss"][-1]
    
    
if __name__ == '__main__':
    from networks.Mlp import Mlp
    from data.Data import Data
    
    mlp = Mlp(3, 5, 5, "relu", 1.0, 1.0)
    data = Data(3, 50)
    
    trainer = GradientFlow(mlp, data, "mse", lr=0.01)
    
    # train
    final_params = trainer.train(mlp.params, num_steps=30000, verbose=True)
    
    # Check final loss
    final_loss = trainer.get_current_loss()
    print(f"Final loss: {final_loss}")