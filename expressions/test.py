import sympy as sp
import jax.numpy as jnp
import jax
import numpy as np

def compute_ntk(k:int, d:int, m:int, h:int):

    if k == 2 :

        # SYMBOLS INITIALIZATION

        # intermediate outputs
        intermediates_l = []
        xl_0 = sp.Matrix(sp.symbols('xl_0_1:%d' % (d + 1)))

        for i in range(1, h + 1):
            xl_i = sp.Matrix(sp.symbols(f'xl_{i}_1:%d' % (h + 1)))
            intermediates_l.append(xl_i)
        
        intermediates_r = []
        xr_0 = sp.Matrix(sp.symbols('xr_0_1:%d' % (d + 1)))

        for i in range(1, h + 1):
            xr_i = sp.Matrix(sp.symbols(f'xr_{i}_1:%d' % (h + 1)))
            intermediates_r.append(xr_i)

        # params
        a = sp.Matrix(sp.symbols('a_1:%d' % (m + 1)))
        
        params = []

        W1 = sp.Matrix([[sp.Symbol(f'W_1_{i}_{j}') for j in range(d)] for i in range(m)])
        params.append(W1)

        for i in range(2, h + 1):
            Wi = sp.Matrix([[sp.Symbol(f'W_{i}_{r}_{c}') for c in range(m)] for r in range(m)])
            params.append(Wi)
        
        a = sp.Matrix(sp.symbols('a_1:%d' % (m + 1)))
        params.append(a)

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
        
        for layer in range(h - 1, -1, -1):

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

    else :

        #TODO

        pass
            
    return ntk


def sub_and_compute(expr, params, input_l, input_r, intermediates_l, intermediates_r, activation_prime):

    print("Computing the dictionary of substitutions...")

    subs_dict = {}

    # Ws
    for i, param in enumerate(params[:-1]):
        for row in range(param.shape[0]):
            for col in range(param.shape[1]):
                symbol = sp.Symbol(f'W_{i + 1}_{row}_{col}')
                subs_dict[symbol] = float(param[row, col])
    

    # a
    m = params[0].shape[0]
    a_symbols = sp.symbols('a_1:%d' % (m + 1))
    a_values = params[-1].flatten()
    for i, (symbol, value) in enumerate(zip(a_symbols, a_values)):
        subs_dict[symbol] = float(value)

    # inputs
    d = input_l.shape[0]
    xl_0_symbols = sp.symbols('xl_0_1:%d' % (d + 1))
    for i, (symbol, value) in enumerate(zip(xl_0_symbols, input_l.flatten())):
        subs_dict[symbol] = float(value)
    
    xr_0_symbols = sp.symbols('xr_0_1:%d' % (d + 1))
    for i, (symbol, value) in enumerate(zip(xr_0_symbols, input_r.flatten())):
        subs_dict[symbol] = float(value)
    
    for i, intermediate in enumerate(intermediates_l):
        xl_i_symbols = sp.symbols(f'xl_{i + 1}_1:%d' % (m + 1)) 
        for (symbol, value) in zip(xl_i_symbols, intermediate.flatten()):
            subs_dict[symbol] = float(value)
    
    for i, intermediate in enumerate(intermediates_r):
        xr_i_symbols = sp.symbols(f'xr_{i + 1}_1:%d' % (m + 1)) 
        for symbol, value in zip(xr_i_symbols, intermediate.flatten()):
            subs_dict[symbol] = float(value)


    sqrt_m_sym = sp.Symbol('sqrt_m')
    subs_dict[sqrt_m_sym] = float(np.sqrt(m))

    print("Finished computing the dictionary of substitutions; Substituting...")

    # substitute
    expr = expr.xreplace(subs_dict)

    print("Finished substituting; computing the expression's numerical value...")

    num_expr = sp.lambdify([], expr, modules=[{"sigma_prime": activation_prime}, "jax"])
    val = num_expr()

    print("Finished computing the expression's numerical value")
    return val


def test():
    from networks.Mlp import Mlp
    mlp = Mlp(3, 5, 5, "relu", 1.0, 1.0)
    x_jax = jnp.array([[2.0], [3.0], [4.0]])
    y_jax = jnp.array([[-1.0], [5.0], [2.0]])
    x = np.array([2.0, 3.0, 4.0])
    y = np.array([-1.0, 5.0, 2.0])
    _, intermediates_l = mlp.forward_with_outputs(x_jax)
    _, intermediates_r = mlp.forward_with_outputs(y_jax)
    print("Beginning expression computing...")
    ntk_expr = compute_ntk(2, 3, 5, 5)
    print("Finished expression computing")
    print(sub_and_compute(ntk_expr, mlp.params, x, y, intermediates_l, intermediates_r, mlp.activation_prime))

test()

