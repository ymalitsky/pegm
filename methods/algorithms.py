from __future__ import division
import numpy as np
import scipy as sp
import numpy.linalg as LA
from itertools import count
from time import time

# make random generator fixed for the reproducible research
gen = 123


def initial_lambda(F, x0, a, prec=1e-10):
    """
    Finds initial stepsize.

    With given map F and starting point x0 compute la_0 as an
    approximation of local Lipschitz constant of F in x0. This helps to
    make a better choice for the initial stepsize.  The resulting
    output is the full iterate information for the algorithm.

    """
    np.random.seed(gen)
    x1 = x0 + np.random.random(x0.shape) * prec
    Fx0 = F(x0)
    # need to fix division in case of zero in the denominator
    la0 = a * np.sqrt(np.vdot(x1 - x0, x1 - x0) /
                      np.vdot(F(x1) - F(x0), F(x1) - F(x0)))
    res = [x1, x0, x0, la0, Fx0]
    return res


def alg_VI_prox(J, F, prox, x0, numb_iter=100):
    """
    Implementation of the Algorithm 2 from the paper. 

    Parameters
    ----------
    J: function that checks the progress of iterates. It may be
    ||x-x*||, or a gap function for VI, or just f(x)+g(x) for
    minimization problem. Takes x0-like array as input and gives
    scalar value.
    F: operator in the VI (or a gradient in minimization problems).
    prox: proximal operator prox_g. Takes x-array and positive scalar
    value rho and gives another x-array. 
    x0: array. Initial point.
    numb_iter: positive integer, optional. Number of iteration. In
    general should be replaced by some stopping criteria.

    Returns:
    list of 3 elements: [values, x1, n_F]. Where
    values collect information for comparison.
    """
    begin = time()
    #######  Initialization  ########
    a = 0.41
    tau = 1.618
    sigma = 0.7
    la_max = 1e7
    iterates = [[J(x0)]] + initial_lambda(F, x0, a) + [tau, 2]
    #######

    #######  Main iteration  #######
    def T(values, x, x_old, y_old, la_old, Fy_old, tau_old, n_F):
        tau = np.sqrt(1 + tau_old)
        for j in count(0):
            y = x + tau * (x - x_old)
            Fy = F(y)
            la = tau * la_old
            # if la**2*np.vdot(Fy-Fy_old, Fy-Fy_old) <= a**2*np.vdot(y-y_old,
            # y-y_old):
            if la * LA.norm(Fy - Fy_old) <= a * LA.norm(y - y_old):
                break
            else:
                tau *= sigma
        # print la
        x1 = prox(x - la * Fy, la)
        n_F += j + 1
        values.append(J(x1))
        res = [values, x1, x, y, la, Fy, tau, n_F]
        return res

    # Running x_n+1 = T(x_n)
    for i in xrange(numb_iter):
        iterates = T(*iterates)
    end = time()
    print "---- Alg. 2 ----"
    print "Number of iterations:", numb_iter
    print "Number of gradients, n_grad:", iterates[-1]
    print "Number of prox_g:", numb_iter
    print "Time execution:", end - begin
    return [iterates[i] for i in [0, 1, -1]]


def alg_VI_prox_affine(J, F, prox, x0, numb_iter=100):
    """
    Implementation of the Algorithm 2 from the paper in case when F is affine. 
    The docs are the same as for the function alg_VI_prox.
    """
    begin = time()
    #######  Initialization  ########
    a = 0.41
    tau = 1.618
    sigma = 0.7
    init = initial_lambda(F, x0, a)
    iterates = [[J(x0)]] + init + [init[-1]] + [tau, 2]
    #######

    #######  Main iteration  #######
    def T(values, x, x_old, y_old, la_old, Fx_old, Fy_old, tau_old, n_F):
        tau = np.sqrt(1 + tau_old)
        Fx = F(x)
        for j in count(0):
            y = x + tau * (x - x_old)
            Fy = (1 + tau) * Fx - tau * Fx_old
            la = tau * la_old
            if la**2 * np.vdot(Fy - Fy_old, Fy - Fy_old) <= a**2 * np.vdot(y - y_old, y - y_old):
                break
            else:
                tau *= sigma
        x1 = prox(x - la * Fy, la)
        n_F += 1
        values.append(J(x1))
        res = [values, x1, x, y, la, Fx, Fy, tau, n_F]
        return res

    # Running x_n+1 = T(x_n)
    for i in xrange(numb_iter):
        iterates = T(*iterates)

    end = time()
    print "---- Alg. 2, affine operator ----"
    print "Number of iterations:", numb_iter
    print "Number of gradients, n_grad:", iterates[-1]
    print "Number of prox_g:", numb_iter
    print "Time execution:", end - begin
    return [iterates[i] for i in [0, 1, -1]]


#------------------------

def find_lambda(a, b, c2, upper_bound):
    """
    Solves quadratic equation ||x*a-b||**2 <= c
    If possible returns a pair of real roots and (0,0) otherwise
    """
    aa = np.vdot(a, a)
    bb = np.vdot(b, b)
    ab = np.vdot(a, b)
    if aa == 0:
        res = (0, upper_bound) if bb <= c else (0, 0)
    else:
        D = ab**2 - aa * (bb - c2)
        if D >= 0:
            la_1 = (ab + np.sqrt(D)) / aa
            la_0 = (ab - np.sqrt(D)) / aa
            res = (la_0, la_1)
        else:
            res = (0, 0)
    return res


def alg_VI_proj(J, F, prox, x0, tau_0=1, constr='True',  numb_iter=100):
    """
    Implementation of the Algorithm 1 from the paper. 

    Parameters 
    ---------- 
    J: function that checks the progress of iterates. It may be
    ||x-x*||, or a gap function for VI, or just f(x)+g(x) for
    minimization problem. Takes x0-like array as input and gives
    scalar value.  
    F: operator in the VI (or a gradient in minimization problems).
    prox: proximal operator prox_g. Takes x-array and positive scalar
    value rho and gives another x-array. Actually for convergence this
    operator should be a projection onto some convex closed
    set. Otherwise, it may disconverge.
    x0: array. Initial point. 
    tau_0: a real number,  optional.
    constr: = False if the set of constraints is affine (this includes
    empty set). True otherwise.
    numb_iter: positive integer, optional. Number of iteration. In
    general should be replaced by some stopping criteria.

    Returns:
    list of 3 elements: [values, x1, n_F]. Where values collect
    information for comparison.  Works only when prox is a projection
    mapping

    """
    begin = time()
    a = 0.41
    sigma = 0.9
    la_max = 1000
    iterates = [[J(x0)]] + initial_lambda(F, x0, a) + [tau_0, 2]

    def T(values, x, x_old, y_old, la_old, Fy_old, tau_old, n_F):
        #tau = tau_0
        tau = 1. / sigma
        for j in count(0):
            y = x + tau * (x - x_old)
            Fy = F(y)
            b = tau * la_old * Fy_old
            c2 = a**2 * np.vdot(y - y_old, y - y_old)
            up_bound = (1 + tau_old) / tau * la_old if constr else 0.5 * la_max
            la0, la1 = find_lambda(Fy, b, c2, up_bound)
            if la1 > 0 and (la1 <= up_bound or la0 <= up_bound):
                la = min(la1, up_bound)
                break
            else:
                tau *= sigma
                # print j, Fy, y

        print la, tau, LA.norm(2 * la_old * Fy - la_old * Fy_old) - a * LA.norm(y - y_old)
        x1 = prox(x - la * Fy, la)
        n_F += j + 1
        values.append(J(x1))
        res = [values, x1, x, y, la, Fy, tau, n_F]
        return res

    for i in xrange(numb_iter):
        iterates = T(*iterates)

    end = time()
    print "---- Alg. 1 ----"
    print "Number of iterations:", numb_iter
    print "Number of gradients, n_grad:", iterates[-1]
    print "Number of prox_g:", numb_iter
    print "Time execution:", end - begin
    return [iterates[i] for i in [0, 1, -1]]


def alg_VI_proj_affine(J, F, prox, x0, tau_0=1, constr='True',  numb_iter=100):
    """
    Implementation of the Algorithm 1 from the paper in case when F is affine. 
    The docs are the same as for the function alg_VI_proj.
    """
    begin = time()
    a = 0.41
    sigma = 0.7
    la_max = 1000
    init = initial_lambda(F, x0, a)
    iterates = [[J(x0)]] + init + [init[-1]] + [tau_0, 2]

    def T(values, x, x_old, y_old, la_old, Fx_old, Fy_old, tau_old, n_F):
        tau = tau_0
        Fx = F(x)
        for j in count(0):
            y = x + tau * (x - x_old)
            Fy = (1 + tau) * Fx - tau * Fx_old
            b = tau * la_old * Fy_old
            c2 = a**2 * np.vdot(y - y_old, y - y_old)
            up_bound = (1 + tau_old) / tau * la_old if constr else 0.5 * la_max
            la0, la1 = find_lambda(Fy, b, c2, up_bound)
            if la1 > 0 and (la1 <= up_bound or la0 <= up_bound):
                la = min(la1, up_bound)
                break
            else:
                tau *= sigma
        x1 = prox(x - la * Fy, la)
        n_F += 1
        values.append(J(x1))
        res = [values, x1, x, y, la, Fx, Fy, tau, n_F]
        return res

    for i in xrange(numb_iter):
        iterates = T(*iterates)

    end = time()
    print "---- Alg. 1, affine operator ----"
    print "Number of iterations:", numb_iter
    print "Number of gradients, n_grad:", iterates[-1]
    print "Number of prox_g:", numb_iter
    print "Time execution:", end - begin
    return [iterates[i] for i in [0, 1, -1]]


def alg_VI_prox_minim(J, F, prox, x0, tau0=1, theta=2, numb_iter=100):
    """
    Implementation of the Algorithm 3 from the paper (for the
    composite minimization problem: min f(x) + g(x))

    Parameters
    ----------
    J: function that checks the progress of iterates. It may be
    ||x-x*||, or just f(x)+g(x) for minimization problem. Takes
    x0-like array as input and gives scalar value.
    F: gradient of f.
    prox: proximal operator prox_g. Takes x-array and positive scalar
    value rho and gives another x-array. 
    x0: array. Initial point.
    tau0: real positive number , optional
    theta: real number from range [1,2], optional
    numb_iter: positive integer, optional. Number of iteration. In
    general should be replaced by some stopping criteria.

    Returns:
    list of 3 elements: [values, x1, n_F]. Where
    values collect information for comparison.
    """
    begin = time()
    tau = tau0
    a = 0.41 * (2 - 1. / theta)
    sigma = 0.7
    iterates = [[J(x0)]] + initial_lambda(F, x0, a) + [tau, 2]

    def T(values, x, x_old, y_old, la_old, Fy_old, tau_old, n_F):
        tau = np.sqrt((1 + theta * tau_old) / (2 * theta - 1))
        for j in count(0):
            y = x + tau * (x - x_old)
            Fy = F(y)
            la = (2 - 1. / theta) * tau * la_old
            if la**2 * np.vdot(Fy - Fy_old, Fy - Fy_old) <= a**2 * np.vdot(y - y_old, y - y_old):
                break
            else:
                tau *= sigma

        x1 = prox(x - la * Fy, la)
        n_F += j + 1
        values.append(J(x1))
        res = [values, x1, x, y, la, Fy, tau, n_F]
        return res

    for i in xrange(numb_iter):
        iterates = T(*iterates)

    end = time()
    print "---- Alg. 3 ----"
    print "Number of iterations:", numb_iter
    print "Number of gradients, n_grad:", iterates[-1]
    print "Number of prox_g:", numb_iter
    print "Time execution:", end - begin
    return [iterates[i] for i in [0, 1, -1]]


#########
def fista_linesearch(J, f, d_f, prox_g, x0, la0=1, numb_iter=100):
    """
    Minimize function F(x) = f(x) + g(x) by FISTA with backtracking.
    Takes J as some evaluation function for comparison.
    """
    begin = time()
    values = [J(x0)]
    iterates = [values, x0, x0, 1, la0, 0, 0, 0]

    def iter_T(values, x, y, t, la, n_f, n_df, n_prox):
        dfy = d_f(y)
        fy = f(y)
        sigma = 0.7
        for j in count(0):
            x1 = prox_g(y - la * dfy, la)
            if f(x1) <= fy + np.vdot(dfy, x1 - y) + 0.5 / la * LA.norm(x1 - y)**2:  # F(x1) < Q(x1, y)
                break
            else:
                la *= sigma
        # print la
        # if n_prox == 0:
        #    print la
        t1 = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
        y1 = x1 + (t - 1) / t1 * (x1 - x)
        n_f += j + 2
        n_df += j + 1
        n_prox += j + 1
        values.append(J(x1))
        ans = [values, x1, y1, t1, la, n_f, n_df, n_prox]
        return ans

    for i in xrange(numb_iter):
        iterates = iter_T(*iterates)

    end = time()
    print "---- FISTA ----"
    print "Number of iterations:", numb_iter
    print "Number of function, n_f:", iterates[-3]
    print "Number of gradients, n_grad:", iterates[-2]
    print "Number of prox_g:", iterates[-1]
    print "Time execution:", end - begin
    return [iterates[i] for i in [0, 1, -3, -2, -1]]


def prox_grad_linesearch(J, f, d_f, prox_g, x0, la0=1, numb_iter=100):
    """
    Minimize function F(x) = f(x) + g(x) by proximal gradient method
    with backtracking.  Takes J as some evaluation function for
    comparison.
    """
    begin = time()
    values = [J(x0)]
    fx0 = f(x0)
    iterates = [values, x0, fx0, la0, 0, 0, 0]

    def iter_T(values, x, fx, la, n_f, n_df, n_prox):
        dfx = d_f(x)
        sigma = 0.7
        for j in count(0):
            x1 = prox_g(x - la * dfx, la)
            fx1 = f(x1)
            if fx1 <= fx + np.vdot(dfx, x1 - x) + 0.5 / la * LA.norm(x1 - x)**2:
                break
            else:
                la *= sigma
        # print la
        values.append(J(x1))
        # if n_prox == 0:
        #    print la
        n_f += j + 1
        n_df += j + 1
        n_prox += j + 1
        ans = [values, x1, fx1,  1.5 * la, n_f, n_df, n_prox]
        return ans

    for i in xrange(numb_iter):
        iterates = iter_T(*iterates)\

    end = time()
    print "---- PGM ----"
    print "Number of iterations:", numb_iter
    print "Number of function, n_f:", iterates[-3]
    print "Number of gradients, n_grad:", iterates[-2]
    print "Number of prox_g:", iterates[-1]
    print "Time execution:", end - begin
    return [iterates[i] for i in [0, 1, -3, -2, -1]]


def tseng_fbf_linesearch(J, F, prox_g, x0, delta=2, numb_iter=100):
    """
    Solve monotone inclusion $0 \in F + \partial g by Tseng
    forward-backward-forward algorithm with backtracking. In
    particular, minimize function F(x) = f(x) + g(x) with convex
    smooth f and convex g. Takes J as some evaluation function for
    comparison.

    """
    begin = time()
    beta = 0.7
    theta = 0.99
    la0 = initial_lambda(F, x0, theta)[3]
    iterates = [[J(x0)], x0, la0, 1, 0]

    def iter_T(values, x, la, n_F, n_prox):
        Fx = F(x)
        la *= delta
        for j in count(0):
            z = prox_g(x - la * Fx, la)
            Fz = F(z)
            if la * LA.norm(Fz - Fx) <= theta * LA.norm(z - x):
                break
            else:
                la *= beta
        x1 = z - la * (Fz - Fx)
        # print j, la
        values.append(J(z))
        # n_f += j+1
        n_F += j + 2
        n_prox += j + 1
        ans = [values, x1, la,  n_F, n_prox]
        return ans

    for i in xrange(numb_iter):
        iterates = iter_T(*iterates)

    end = time()
    print "---- FBF ----"
    print "Number of iterations:", numb_iter
    print "Number of gradients, n_grad:", iterates[-2]
    print "Number of prox_g:", iterates[-1]
    print "Time execution:", end - begin
    return [iterates[i] for i in [0, 1,  -2, -1]]


def extragradient_iusem_svaiter(J, F, proj, x0, delta=2., sigma=0.5, numb_iter=100):
    """
    Solve variational inequality <F(x),y-x> >=0 using extragradient
    method with the linesearch of Iusem-Svaiter.

    """
    begin = time()
    beta = 0.7
    #sigma = 0.5
    la0 = initial_lambda(F, x0, 1)[3]
    iterates = [[J(x0)], x0, 1, 0]
    la = delta * la0

    def iter_T(values, x, n_F, n_prox):
        Fx = F(x)
        y = proj(x - la * Fx)
        t = 1
        for j in count(0):
            z = t * y + (1. - t) * x
            Fz = F(z)
            if la * np.dot(Fz, x - y) >= sigma * np.dot(x - y, x - y):
                break
            else:
                t *= beta
        # print j, la

        x1 = proj(x - np.dot(Fz, x - z) / np.dot(Fz, Fz) * Fz)
        values.append(J(z))
        # n_f += j+1
        n_F += j + 2
        n_prox += 2
        ans = [values, x1,  n_F, n_prox]
        return ans

    for i in xrange(numb_iter):
        iterates = iter_T(*iterates)

    end = time()
    print "---- Extragradient + Linesearch of Iusem-Svaiter ----"
    print "Number of iterations:", numb_iter
    print "Number of gradients, n_grad:", iterates[-2]
    print "Number of prox_g:", iterates[-1]
    print "Time execution:", end - begin
    return [iterates[i] for i in [0, 1,  -2, -1]]


def prox_method_korpelevich(J, F, prox, x0, la, numb_iter=100):
    """find a solution VI(F,Set) by Korpelevich method."""
    res = [[J(x0)], x0]

    def iter_T(values, x):
        y = prox(x - la * F(x), la)
        x = prox(x - la * F(y), la)
        values.append(J(x))
        return [values, x]

    for i in xrange(numb_iter):
        res = iter_T(*res)
    return res


def prox_reflected_grad(J, F, prox_g, x0, la, numb_iter=100):
    y = x0
    x = x0
    values = []
    for i in xrange(numb_iter):
        x1 = prox_g(x - la * F(y), la)
        y = 2 * x1 - x
        x = x1
        values.append(J(x))
    return values, x, y


def pock_chamb(J, prox_g1, prox_g2, df, K, sigma, tau, x0, y0, numb_iter=100):
    """ min_x max_y (<Kx,y>  + g1(x) - g2(y))* """
    begin = time()
    theta = 1.0
    x, y, z = x0, y0, x0
    values = [J(x0, y0)]
    for i in xrange(numb_iter):
        x1 = prox_g1(x - tau * (df(x) + K.T.dot(y)), tau)
        z = x1 + theta * (x1 - x)
        y = prox_g2(y + sigma * K.dot(z), sigma)
        x = x1
        values.append(J(x, y))
    end = time()
    print "----- Primal-dual method -----"
    print "Number of iterations:", numb_iter
    print "Number of matrix-vector multiplications:", 2 * numb_iter
    print "Time execution:", end - begin
    return [values, x]


def pock_chamb_linesearch(J, prox_g1, prox_g2, K,  x0, y0, la, beta, theta=1, numb_iter=100):
    """ min_x max_y (<Kx,y>  + g1(x) - g2(y))* """
    begin = time()
    theta = 1.0
    x, y, z = x0, y0, x0
    values = [J(x0, y0)]
    mu = 0.7
    iterates = [values, y0, x0, theta, la, K.dot(x0)]

    def T(values, y, x_old, th_old, la_old, Kx_old):
        x = prox_g1(x_old - la_old / beta * K.T.dot(y), la_old / beta)
        Kx = K.dot(x)
        th = np.sqrt(1 + th_old)
        for j in count(0):
            la = la_old * th
            z = x + th * (x - x_old)
            Kz = Kx + th * (Kx - Kx_old)
            y1 = prox_g2(y + la * beta * K.dot(z), la * beta)
            if la * LA.norm(K.T.dot(y1 - y)) <= 0.9 * LA.norm(y1 - y):
                break
            else:
                th *= mu
        values.append(J(x, y1))
        res = [values, y1, x, th, la, Kx]
        return res

    for i in xrange(numb_iter):
        iterates = T(*iterates)

    end = time()
    print "----- Primal-dual method with linesearch-----"
    print "Number of iterations:", numb_iter
    print "Number of matrix-vector multiplications:", 2 * numb_iter
    print "Time execution:", end - begin
    return [iterates[i] for i in [0, 2, 1]]  # fixed this, previously was wrong


def initial_lambda2(F, h, x0, a, prec=1e-10):
    """
    Finds initial stepsize.

    With given map F and starting point x0 compute la_0 as an
    approximation of local Lipschitz constant of F in x0. This helps to
    make a better choice for the initial stepsize.  The resulting
    output is the full iterate information for the algorithm.

    """
    np.random.seed(gen)
    x1 = x0 + np.random.random(x0.shape) * prec
    Fx0 = h(x0, x1) * F(x0)
    Fx1 = h(x1, x0) * F(x1)
    # need to fix division in case of zero in the denominator
    la0 = a * np.sqrt(np.vdot(x1 - x0, x1 - x0) /
                      np.vdot(Fx1 - Fx0, Fx1 - Fx0))
    res = [x1, x0, x0, la0, Fx0]
    return res


def initial_lambda_variable(F, x0, a, prec=1e-10):
    """
    Finds initial stepsize.

    With given map F and starting point x0 compute la_0 as an
    approximation of local Lipschitz constant of F in x0. This helps to
    make a better choice for the initial stepsize.  The resulting
    output is the full iterate information for the algorithm.

    """
    np.random.seed(gen)
    x1 = x0 + np.random.random(x0.shape) * prec
    Fx0 = F(x0)
    Fx1 = F(x1)
    # need to fix division in case of zero in the denominator
    la0 = a * np.abs((x1 - x0) / (Fx1 - Fx0))
    res = [x1, x0, x0, la0, Fx0]
    return res


def alg_VI_variable(J, F, x0, numb_iter=100):
    begin = time()
    #######  Initialization  ########
    a = 0.5
    tau = 1.618
    sigma = 0.7
    la_max = 1e7
    iterates = [[J(x0)]] + initial_lambda_variable(F, x0, a) + [tau, 2]
    #######

    #######  Main iteration  #######
    def T(values, x, x_old, y_old, la_old, Fy_old, tau_old, n_F):
        tau = np.sqrt(1 + tau_old)
        for j in count(0):
            y = x + tau * (x - x_old)
            Fy = F(y)
            la = tau * la_old
            if np.alltrue(la * np.abs(Fy - Fy_old) <= a * np.abs(y - y_old)):
                # if LA.norm(la * (Fy - Fy_old)) <= a * La.norm(y - y_old):
                break
            else:
                tau *= sigma
        print 'j=', j, tau
        print la
        x1 = x - la * Fy
        n_F += j + 1
        values.append(J(x1))
        res = [values, x1, x, y, la, Fy, tau, n_F]
        return res

    # Running x_n+1 = T(x_n)
    for i in xrange(numb_iter):
        iterates = T(*iterates)
    end = time()
    print "---- Alg. 2 ----"
    print "Number of iterations:", numb_iter
    print "Number of gradients, n_grad:", iterates[-1]
    print "Number of prox_g:", numb_iter
    print "Time execution:", end - begin
    return [iterates[i] for i in [0, 1, -1]]


def alg_VI_variable2(J, F, x0, numb_iter=100):
    begin = time()
    #######  Initialization  ########
    a = 0.5
    d = x0.shape
    tau = 1.618 * np.ones(d)
    sigma = 0.7
    la_max = 1e7
    iterates = [[J(x0)]] + initial_lambda_variable(F, x0, a) + [tau, 2]
    #######

    #######  Main iteration  #######
    def T(values, x, x_old, y_old, la_old, Fy_old, tau_old, n_F):
        tau = np.sqrt(1 + tau_old)
        for j in count(0):
            y = x + tau * (x - x_old)
            Fy = F(y)
            la = tau * la_old
            w = la * np.abs(Fy - Fy_old) <= a * np.abs(y - y_old)
            if np.alltrue(w):
                break
            else:
                w2 = 1 - (1 - w) * (1 - sigma)
                tau = tau * w2
        print 'j=', j, tau
        print la
        x1 = x - la * Fy
        n_F += j + 1
        values.append(J(x1))
        res = [values, x1, x, y, la, Fy, tau, n_F]
        return res

    # Running x_n+1 = T(x_n)
    for i in xrange(numb_iter):
        iterates = T(*iterates)
    end = time()
    print "---- Alg. 2 ----"
    print "Number of iterations:", numb_iter
    print "Number of gradients, n_grad:", iterates[-1]
    print "Number of prox_g:", numb_iter
    print "Time execution:", end - begin
    return [iterates[i] for i in [0, 1, -1]]


def alg_VI_variable3(J, F, x0, tau_0=1, numb_iter=100):
    begin = time()
    a = 0.5
    sigma = 0.7
    la_max = 1000
    d = x0.shape
    tau = tau_0
    iterates = [[J(x0)]] + initial_lambda_variable(F, x0, a) + [tau, 2]

    def T(values, x, x_old, y_old, la_old, Fy_old, tau_old, n_F):
        tau = tau_0
        for j in count(0):
            y = x + tau * (x - x_old)
            Fy = F(y)
            b = tau * la_old * Fy_old
            c2 = (a * (y - y_old))**2
            up_bound = 0.5 * la_max * np.ones(d)
            #la = a * np.abs(y - y_old) / np.abs(Fy - Fy_old)
            la_big = np.array(map(find_lambda, Fy, b, c2, up_bound))
            w = la_big[:, 1] > 0
            # and (np.alltrue(la_big[:, 1] <= up_bound) or np.alltrue(la_big[:,
            # 0] <= up_bound)):
            print w
            if np.alltrue(w):
                la = np.fmin(la_big[:, 1], up_bound)
            # if LA.norm(la * Fy - tau * la_old * Fy_old) <= a * LA.norm(y -
            # y_old):
                break
            else:
                w2 = 1 - (1 - w) * (1 - sigma)
                tau = tau * w2
        print j
        x1 = x - la * Fy
        n_F += j + 1
        values.append(J(x1))
        res = [values, x1, x, y, la, Fy, tau, n_F]
        return res

    for i in xrange(numb_iter):
        iterates = T(*iterates)

    end = time()
    print "---- Alg. 1 ----"
    print "Number of iterations:", numb_iter
    print "Number of gradients, n_grad:", iterates[-1]
    print "Number of prox_g:", numb_iter
    print "Time execution:", end - begin
    return [iterates[i] for i in [0, 1, -1]]


def fb_cruz_nghia1(J, df, prox_g, x0, a0, beta=1.5, numb_iter=100):
    """
    Minimize function F(x) = f(x) + g(x) by proximal gradient method
    with linesearch, proposed in [1] (Method 1)


    [1] J.Cruz, T.Nghia 'On the convergence of the forward-backward
    splitting method with linesearches'

    """

    begin = time()
    th = 0.7
    delta = 0.49
    values = [J(x0)]
    iterates = [values, x0, a0]

    def T(values, x, a):
        df_x = df(x)
        for i in count(0):
            z = prox_g(x - a * df_x, a)
            if a * LA.norm(df(z) - df_x) <= delta * LA.norm(z - x):
                break
            else:
                a *= th
        x = z
        values.append(J(x))
        res = [values, x, beta * a]
        return res

    for i in range(numb_iter):
        iterates = T(*iterates)
    end = time()
    print "---- Forward-backward method 1----"
    print "Number of iterations:", numb_iter
    # print "Number of gradients, n_grad:", iterates[-1]
    # print "Number of prox_g:", numb_iter
    print "Time execution:", end - begin

    return iterates


def fb_cruz_nghia3(J, f, g, df, prox_g, x0, numb_iter=100):
    """
    Minimize function F(x) = f(x) + g(x) by proximal gradient method
    with linesearch, proposed in [1] (Method 3)


    [1] J.Cruz, T.Nghia 'On the convergence of the forward-backward
    splitting method with linesearches'
    """

    begin = time()
    th = 0.7
    delta = 0.49
    values = [J(x0)]
    iterates = [values, x0]

    def T(values, x):
        df_x = df(x)
        z = prox_g(x - df_x, 1)
        beta = 1
        for i in count(0):
            x1 = x - beta * (x - z)
            if f(x1) + g(x1) <= f(x) + g(x) - beta * (g(x) - g(z)) - beta * df_x.dot(x - z) + beta / 2 * LA.norm(x - z)**2:
                break
            else:
                beta *= th
        x = x1
        values.append(J(x))
        res = [values, x]
        return res

    for i in range(numb_iter):
        iterates = T(*iterates)
    end = time()
    print "---- Forward-backward method 3----"
    print "Number of iterations:", numb_iter
    # print "Number of gradients, n_grad:", iterates[-1]
    # print "Number of prox_g:", numb_iter
    print "Time execution:", end - begin

    return iterates
