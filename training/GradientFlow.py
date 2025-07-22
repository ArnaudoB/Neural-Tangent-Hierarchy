import jax
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_leaves
from jax import jit

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
        
        self.compute_loss_params_jit = jit(self._compute_loss_on_params)
        self.compute_gd_step_jit = jit(self._compute_gd_step)
        
        
    def _compute_loss_on_params(self, params, x, y):
        predictions = self.model.forward_with_params(params, x)
        return self.loss_fn(predictions, y)
    
    def compute_loss_on_params(self, params, x, y):
        return self.compute_loss_params_jit(params, x, y)
    
    
    def _compute_gd_step(self, params, x, y, dt):

        loss_grad_fn = jax.grad(self._compute_loss_on_params, argnums=0)
        grads = loss_grad_fn(params, x, y)
        
        new_params = tree_map(lambda p, g: p - dt * g, params, grads)
      
        loss_value = self._compute_loss_on_params(params, x, y)
        grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in tree_leaves(grads)))
        
        return new_params, loss_value, grad_norm
    
    def compute_gd_step(self, params, x, y, dt):
        return self.compute_gd_step_jit(params, x, y, dt)

    
    
    def step(self, params, dt=None):
    
        if dt is None:
            dt = self.lr
        
        new_params, loss_value, grad_norm = self.compute_gd_step(params, self.x, self.y, dt)
        
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
    

    def get_current_loss(self, params):
        return self.compute_loss_on_params(params, self.x, self.y)
    
    
if __name__ == '__main__':
    from networks.Mlp import Mlp
    from data.Data import Data
    
    mlp = Mlp(3, 5, 5, "relu", 1.0, 1.0)
    data = Data(3, 50)
    
    trainer = GradientFlow(mlp, data, "mse", lr=0.01)
    
    # train
    final_params = trainer.train(mlp.params, num_steps=1000000, verbose=True)
    
    # Check final loss
    final_loss = trainer.get_current_loss(final_params)
    print(f"Final loss: {final_loss}")