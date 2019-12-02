import numpy as np
import sympy as sp

# New for this assignment
def array(*args, ndmin = 1):
    return np.array(args, ndmin = ndmin)

# From Homework3
class Function:
    def __init__(self, func_str, symbols, name = '', parameter_pos = None):
        '''
            (1) symbols are passed in as a string and seperated by ','
            (2) parameter_pos remembers which variable the function should take. 
                Default is None because the original function takes all variables.
                It is only useful for its partial derivatives functions.
        '''
        self.func = func_str
        self.symbols = [s.strip() for s in symbols.split(',')]
        self.s = len(self.symbols)
        self.name = name
        exec(f'self._f = lambda {symbols}: {func_str}')
        
        self.pos = parameter_pos
        if not self.pos:
            self.pos = list(range(self.s))       
    
    def __repr__(self):
        return self.func
    
    def __call__(self, variables):
        ''' to call the function, we only take what need in the variable vector '''
        if self.symbols == ['']:
            return eval(self.func)
        if len(variables.shape) == 1: # (n,)
            s = [variables[i] for i in self.pos]
            return self._f(*s)
        elif variables.shape[0] == 1: # (1, n)
            s = [variables[0][i] for i in self.pos]
            return self._f(*s)
        raise AssertionError(f'Dimension dismatch {variables.shape}')
    
    def diff(self, symbol):
        ''' differentiate using sympy, remember all the varibale positions and return a new Function object '''
        df = str(sp.diff(self.func, symbol))  #str(sp.Function(str(sp.diff(self.func, symbol))))
        sym, pos = [], []
        for i, s in enumerate(self.symbols):
            if s in df:
                sym.append(s)
                pos.append(i)
        return Function(df, ','.join(sym), parameter_pos = pos)
    
# From Homework2
def backtracking_line_search(f, f_gradient, start, direction, step_size = 1, p = 0.5, beta = 0.5):
    y, g = f(start), compute_vector(f_gradient, start)
    
    # a step_size alpha is accepted only when f(x + alpha*d) is below the line f(x) + alpha * beta * gradient * direction
    while f(start + step_size * direction) > (y + beta * step_size * (g.dot(direction))):
        step_size *= p # decrease the step size by a factor if not accepetable 
    return step_size

def conjugate_gradients(f : 'Function', x0 : 'np.array', fprime : ['Function'] , tol = 1e-4, b = 0.5):
    x = x0.copy()
    iteration, step, direction = 0, 0, 0 # initialization
   
    # main loop
    gm = gradient_magnitude(fprime, x)
    while  gm > tol: 
        beta, direction = 0, 0 
        while True:
            iteration += 1
            direction = -compute_vector(fprime, x) / gm + beta * direction
            step = backtracking_line_search(f, fprime, x, direction, beta = b)

            x += step * direction
            gm_prev = gm
            gm = gradient_magnitude(fprime, x)
            
            if iteration % 5 == 0: # restart 
                break
            beta = (gm / gm_prev)**2 #gm**2 / gm_prev**2
    return x

# Helpers
def gradient(f:'Function') -> ['Function']:
    ''' return a gradient vector'''
    return np.array([f.diff(s) for s in f.symbols])# (n,)

def compute_vector(v: ['Function'], x: 'np.array', ndmin = 1) -> ['value']:
    ''' compute the gradient value at given point'''
    return np.array([d(x) for d in v], dtype = 'float64', ndmin = ndmin)

def gradient_magnitude(g, x):
    ''' return the length of the gradient vector at point/vector x'''
    if len(g.shape) == 1: # (n,)
        return np.sqrt(sum([i(x)**2 for i in g]))
    elif g.shape[0] == 1: # (1, n)
        return np.sqrt(sum([i(x)**2 for i in g[0]]))
    raise AssertionError(f'Dimension dismatch {g.shape}')
        
def vector_length(v):
    ''' return the length of a vector'''
    if len(v.shape) == 1: # (n,)
        return np.sqrt(sum([i**2 for i in v]))
    elif v.shape[0] == 1: # (1, n)
        return np.sqrt(sum([i**2 for i in v[0]]))
    raise AssertionError(f'Dimension dismatch {v.shape}')
    
'''    
def initial_min_bracket(f, start = 0, step = 1e-2, expand_factor = 2.0):
    a, fa = start, f(start)
    b, fb = start + step, f(start + step)
    if fb > fa:
        a, b = b, a
        fa, fb = fb, fa
        step = -step
    while True:
        c, fc = b + step, f(b + step)
        if fc > fb:
            return (a, c) if a <= c else (c, a)
        a, fa, b, fb = b, fb, c, fc
        step *= expand_factor
        
def golden_section(func, a, b, tol = 1e-6):
    phi = (1 + np.sqrt(5))/2 - 1
    d = phi * (b - a) + a  
    fd = func(d)
    
    while abs(b - a) > tol:
        c = b - phi * (b - a) 
        fc = func(c)
        if fc < fd:
            b, d, fd = d, c, fc
        else:
            a, b = b, c
    return (a + b) / 2

def line_search(f, x, direction):
    objective = lambda alpha : f(x + alpha * direction)
    a, b = initial_min_bracket(objective)
    return golden_section(objective, a, b)

def BFGS_method(Q: 'np.array', delta: 'np.array', gama: 'np.array', dim = 10):
    # do matrix operations
    delta, gama = np.mat(delta), np.mat(gama)
    
    d_g = delta * gama.T
    if np.asscalar(d_g) == 0: # avoid 0 division
        return Q
    d_g_Q = delta.T *(gama) * Q
    Q_g_d = Q * gama.T *(delta)
    g_Q_g = gama * Q * (gama.T)
    d_d = delta.T * (delta)
    
    a = (d_g_Q + Q_g_d) / d_g
    b = d_d/d_g + np.asscalar(g_Q_g) * d_d / d_g**2
    return np.asarray(Q - a + b) # back to array

def quasi_newton(f : 'Function', initial_point : 'np.array', fprime : ['Function'] , method = BFGS_method, tol = 1e-4):
    # initializations
    x = initial_point.copy()
    Q = np.identity(x.shape[1])  # inverse Hessian initially be a identity matrix
    iteration, delta = 0, np.zeros((1,x.shape[1]))
    
    # main loop
    g = compute_vector(fprime, x, ndmin=2)
    while vector_length(g) > tol:  # stopping criterion: gradient magnitude
        iteration += 1
        # update x
        direction = -g.dot(Q)
        step = line_search(f, x, direction)
        delta = step * direction
        x += delta
            
        # update Q, esitimated inverse Hessian
        g_prev = g
        g = compute_vector(fprime, x, ndmin=2)
        gama = g - g_prev
        Q = method(Q, delta, gama, dim)
    return x
'''
if __name__ == '__main__':
	pass
