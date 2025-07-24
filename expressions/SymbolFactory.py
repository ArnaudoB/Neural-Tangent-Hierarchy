import sympy as sp

class SymbolFactory:
    """Creates symbolic variables for NTK expressions."""
    
    def __init__(self, d: int, m: int, h: int):
        self.d = d
        self.m = m
        self.h = h
    
    def create_input_symbols(self):
        """Create symbolic input variables."""

        xl_0 = sp.Matrix(sp.symbols(f'xl_0_1:{self.d + 1}'))
        xr_0 = sp.Matrix(sp.symbols(f'xr_0_1:{self.d + 1}'))
        return xl_0, xr_0
    
    def create_intermediate_symbols(self):
        """Create symbolic intermediate variables."""
        intermediates_l = []
        intermediates_r = []
        
        for i in range(1, self.h + 1):
            xl_i = sp.Matrix(sp.symbols(f'xl_{i}_1:{self.m + 1}'))
            xr_i = sp.Matrix(sp.symbols(f'xr_{i}_1:{self.m + 1}'))
            intermediates_l.append(xl_i)
            intermediates_r.append(xr_i)
        
        return intermediates_l, intermediates_r

    def create_parameter_symbols(self):
        """Create symbolic parameter variables."""
        params = []
        
        # input layer
        W1 = sp.Matrix([[sp.Symbol(f'W_1_{i}_{j}') for j in range(self.d)] 
                       for i in range(self.m)])
        params.append(W1)
        
        # hidden layers
        for i in range(2, self.h + 1):
            Wi = sp.Matrix([[sp.Symbol(f'W_{i}_{r}_{c}') for c in range(self.m)] 
                           for r in range(self.m)])
            params.append(Wi)
        
        # output coefficients
        a = sp.Matrix(sp.symbols(f'a_1:{self.m + 1}'))
        params.append(a)
        
        return params