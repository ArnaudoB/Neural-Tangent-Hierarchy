import sympy as sp
import numpy as np

class ExpressionEvaluator:
    """Handles numerical evaluation of symbolic expressions."""
    
    def __init__(self, d: int, m: int, h: int):
        self.d = d
        self.m = m
        self.h = h
    
    def evaluate(self, expr, model_params, inputs_l, inputs_r, intermediates_l, intermediates_r, activation_prime):
        """Main evaluation method."""
        subs_dict = self._build_substitution_dict(
            model_params, inputs_l, inputs_r, intermediates_l, intermediates_r
        )
        # substitute values
        expr = expr.xreplace(subs_dict)
        
        # convert to numerical function
        return self._evaluate_numerical(expr, activation_prime)
    
    
    def _build_substitution_dict(self, model_params, inputs_l, inputs_r, intermediates_l, intermediates_r):
        """Build dictionary mapping symbols to numerical values."""
        subs_dict = {}
        
        # add parameters
        subs_dict.update(self._substitute_parameters(model_params))
        
        # add inputs
        subs_dict.update(self._substitute_inputs(inputs_l, inputs_r))

        # add intermediates
        subs_dict.update(self._substitute_intermediates(intermediates_l, intermediates_r))

        # add constants
        subs_dict[sp.Symbol('sqrt_m')] = float(np.sqrt(self.m))
        return subs_dict
    
    
    def _substitute_parameters(self, model_params):
        """Substitute parameter values."""
        subs_dict = {}
        
        # Ws
        for i, param in enumerate(model_params[:-1]):
            for row in range(param.shape[0]):
                for col in range(param.shape[1]):
                    symbol = sp.Symbol(f'W_{i + 1}_{row}_{col}')
                    subs_dict[symbol] = float(param[row, col])
        
        # a
        a_symbols = sp.symbols(f'a_1:{self.m + 1}')
        a_values = model_params[-1].flatten()
        for symbol, value in zip(a_symbols, a_values):
            subs_dict[symbol] = float(value)
        
        return subs_dict
    
    def _substitute_inputs(self, input_l, input_r):
        """Substitute input values."""
        subs_dict = {}
        d = input_l.shape[0]
        xl_0_symbols = sp.symbols('xl_0_1:%d' % (d + 1))
        for symbol, value in zip(xl_0_symbols, input_l.flatten()):
            subs_dict[symbol] = float(value)
        
        xr_0_symbols = sp.symbols('xr_0_1:%d' % (d + 1))
        for symbol, value in zip(xr_0_symbols, input_r.flatten()):
            subs_dict[symbol] = float(value)
        
        return subs_dict
    
    
    def _substitute_intermediates(self, intermediates_l, intermediates_r):
        """Substitute intermediate values."""
        subs_dict = {}
        for i, intermediate in enumerate(intermediates_l):
            xl_i_symbols = sp.symbols(f'xl_{i + 1}_1:%d' % (self.m + 1)) 
            for (symbol, value) in zip(xl_i_symbols, intermediate.flatten()):
                subs_dict[symbol] = float(value)
        
        for i, intermediate in enumerate(intermediates_r):
            xr_i_symbols = sp.symbols(f'xr_{i + 1}_1:%d' % (self.m + 1)) 
            for symbol, value in zip(xr_i_symbols, intermediate.flatten()):
                subs_dict[symbol] = float(value)
        
        return subs_dict

    
    def _evaluate_numerical(self, expr, activation_prime):
        """Convert to numerical function and evaluate."""
        num_expr = sp.lambdify([], expr, modules=[{
            "sigma_prime": activation_prime,
        }, "jax"])
        
        return float(num_expr())