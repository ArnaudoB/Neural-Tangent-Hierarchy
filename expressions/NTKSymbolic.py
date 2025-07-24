import sympy as sp
import jax.numpy as jnp
import numpy as np
from SymbolFactory import SymbolFactory
from ExpressionEvaluator import ExpressionEvaluator

class NTKSymbolic:
    """Handles symbolic NTK expressions and computations."""
    
    def __init__(self, d: int, m: int, h: int):
        self.d = d
        self.m = m  
        self.h = h
        self._expression_cache = {}  # Better name than exprs
        self._symbol_factory = SymbolFactory(d, m, h)  # Delegate symbol creation
    
    def get_ntk_expression(self, k: int):
        """Get cached or compute NTK expression for kernel k."""
        if k in self._expression_cache:
            return self._expression_cache[k]
        
        expr = self._compute_ntk_expression(k)
        self._expression_cache[k] = expr
        return expr
    
    def _compute_ntk_expression(self, k: int):

        assert k >= 2, "Kernels indexing starts at 2"

        if k == 2:

            # SYMBOLS INITIALIZATION

            # inputs
            xl_0, xr_0 = self._symbol_factory.create_input_symbols()

            # intermediate outputs
            intermediates_l, intermediates_r = self._symbol_factory.create_intermediate_symbols() 

            # params
            params = self._symbol_factory.create_parameter_symbols()

            # others
            sqrt_m_sym = sp.Symbol('sqrt_m')
            activation_prime = sp.Function("sigma_prime") 


            # EXPRESSION COMPUTATION

            ntk = intermediates_l[-1].dot(intermediates_r[-1]) # G^(H + 1)

            # initialize prefixes

            z_l = params[-2] * intermediates_l[-2]
            v_l = sp.Matrix([activation_prime(z_ij) for z_ij in z_l])

            z_r = params[-2] * intermediates_r[-2]
            v_r = sp.Matrix([activation_prime(z_ij) for z_ij in z_r])
            
            prefix_l = sp.diag(*v_l) * params[-1] / sqrt_m_sym 
            prefix_r = sp.diag(*v_r) * params[-1] / sqrt_m_sym
            
            for layer in range(self.h - 1, -1, -1):

                if layer > 0:
                    ntk += (prefix_l.dot(prefix_r)) * (intermediates_l[layer-1].dot(intermediates_r[layer-1]))
                    # update prefixes 
                    if layer == 1:
                        prev_layer_l = xl_0
                        prev_layer_r = xr_0
                    else:
                        prev_layer_l = intermediates_l[layer-2] 
                        prev_layer_r = intermediates_r[layer-2]
                    
                    z_l = params[layer-1] * prev_layer_l
                    v_l = sp.Matrix([activation_prime(z_ij) for z_ij in z_l])

                    z_r = params[layer-1] * prev_layer_r
                    v_r = sp.Matrix([activation_prime(z_ij) for z_ij in z_r])

                    prefix_l = sp.diag(*v_l) * params[layer].T * prefix_l / sqrt_m_sym
                    prefix_r = sp.diag(*v_r) * params[layer].T * prefix_r / sqrt_m_sym
                else:
                    ntk += (prefix_l.dot(prefix_r)) * (xl_0.dot(xr_0)) # input layer
            
            return ntk
    

    def evaluate_expression(self, expr, model_params, inputs_l, inputs_r, intermediates_l, intermediates_r, activation_prime):
        """Evaluate symbolic expression with numerical values."""
        evaluator = ExpressionEvaluator(self.d, self.m, self.h)
        return evaluator.evaluate(expr, model_params, inputs_l, inputs_r, 
                                intermediates_l, intermediates_r, activation_prime)


if __name__ == '__main__':
    from networks.Mlp import Mlp
    mlp = Mlp(6, 10, 4, "relu", 1.0, 1.0)
    expr = NTKSymbolic(6, 10, 4)
    x_jax = jnp.array([[2.0], [3.0], [4.0], [5.0], [6.0], [7.0]])
    y_jax = jnp.array([[-1.0], [5.0], [2.0], [5.0], [6.0], [7.0]])
    x = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    y = np.array([-1.0, 5.0, 2.0, 5.0, 6.0, 7.0])
    _, intermediates_l = mlp.forward_with_outputs(x_jax)
    _, intermediates_r = mlp.forward_with_outputs(y_jax)
    print("Beginning expression computing...")
    ntk_expr = expr.get_ntk_expression(2)
    print("Finished expression computing")
    print(expr.evaluate_expression(ntk_expr, mlp.params, x, y, intermediates_l, intermediates_r, mlp.activation_prime))
    print(mlp.ntk_autodiff(x_jax, y_jax))