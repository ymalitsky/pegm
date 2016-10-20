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


def alg_VI_prox(F, prox, x0, numb_iter=100, tol=1e-6):
    """
    Implementation of the Algorithm 2 from the paper. 

    Parameters
    ----------
    F: operator in the VI (or a gradient in minimization problems).
    prox: proximal operator prox_g. Takes x-array and positive scalar
    value rho and gives another x-array. 
    x0: array. Initial point.
    numb_iter: positive integer, optional. Number of iteration. 
    tol: stopping tolerance, the algortithm terminates when
    ||x1-y||+||y-x|| <= tol

    Returns:
    list of 2 elements: [values, x1]. Values collects the history of
    all err during all iterations.
    """
    begin = time()
    #######  Initialization  ########
    a = 0.41
    tau = 1.618
    sigma = 0.7
    la_max = 1e7
    err0 = 1
    values = []
    iterates = [values] + initial_lambda(F, x0, a) + [tau, 2, 1]
    #######

    #######  Main iteration  #######
    def T(values, x, x_old, y_old, la_old, Fy_old, tau_old, n_F, err):
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
        err = LA.norm(x1 - y) + LA.norm(y - x)
        values.append(err)
        res = [values, x1, x, y, la, Fy, tau, n_F, err]
        return res

    # Running x_n+1 = T(x_n)
    for i in xrange(numb_iter):
        iterates = T(*iterates)
        err = iterates[-1]
        if err <= tol:
            end = time()
            n_F = iterates[-2]
            print "---- Alg. 2 ----"
            print "Number of iterations:", i + 1
            print "Number of prox_g:", i + 1
            print "Number of F, :", n_F
            print "Time execution:", np.round(end - begin, 2)
            break
    if err > tol:
        print "alg_VI_prox does not terminate after", numb_iter, "iterations"

    return [iterates[i] for i in [0, 1]]


def alg_VI_prox_affine(F, prox, x0, numb_iter=100, tol=1e-6):
    """
    Implementation of the Algorithm 2 from the paper in case when F is affine. 
    The docs are the same as for the function alg_VI_prox.
    """
    begin = time()
    #######  Initialization  ########
    a = 0.41
    tau = 1.618
    sigma = 0.7
    values = []
    init = initial_lambda(F, x0, a)
    iterates = [values] + init + [init[-1]] + [tau, 2, 1]
    #######

    #######  Main iteration  #######
    def T(values, x, x_old, y_old, la_old, Fx_old, Fy_old, tau_old, n_F, err):
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
        err = LA.norm(x1 - y) + LA.norm(y - x)
        values.append(err)
        res = [values, x1, x, y, la, Fx, Fy, tau, n_F, err]
        return res

    # Running x_n+1 = T(x_n)
    for i in xrange(numb_iter):
        iterates = T(*iterates)
        err = iterates[-1]
        if err <= tol:
            end = time()
            n_F = iterates[-2]
            print "---- Alg. 2 ----"
            print "Number of iterations:", i + 1
            print "Number of prox_g:", i + 1
            print "Number of F, :", n_F
            print "Time execution:", np.round(end - begin, 2)
            break
    if err > tol:
        print "alg_VI_prox_affine does not terminate after", numb_iter, "iterations"

    return [iterates[i] for i in [0, 1]]


#------------------------

def find_lambda(a, b, c, upper_bound):
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
        D = ab**2 - aa * (bb - c)
        if D >= 0:
            la_1 = (ab + np.sqrt(D)) / aa
            la_0 = (ab - np.sqrt(D)) / aa
            res = (la_0, la_1)
        else:
            res = (0, 0)
    return res


def alg_VI_proj(F, prox, x0, tau_0=1, constr='True',  numb_iter=100, tol=1e-6):
    """
    Implementation of the Algorithm 1 from the paper.

    Parameters 
    ---------- 
    F: operator in the VI (or a gradient in minimization problems).
    prox: proximal operator prox_g. Takes x-array and positive scalar
    value rho and gives another x-array. Actually for convergence this
    operator should be a projection onto some convex closed
    set. Otherwise, the method may disconverge.
    x0: array. Initial point. 
    tau_0: a real number,  optional.
    constr: = False if the set of constraints is affine (this includes
    empty set). True otherwise.
    numb_iter: positive integer, optional. Number of iteration. In
    general should be replaced by some tolping criteria.
    tol: stopping tolerance, the algortithm terminates when
    ||x1-y||+||y-x|| <= tol


    Returns:
    list of 2 elements: [values, x1]. Where values collects the
    history of all err during all iterations.  information for
    comparison.  

    """
    begin = time()
    a = 0.41
    sigma = 0.7
    la_max = 1000
    values = []
    iterates = [values] + initial_lambda(F, x0, a) + [tau_0, 2, 1]

    def T(values, x, x_old, y_old, la_old, Fy_old, tau_old, n_F, err):
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
        # print la, j
        # print la, tau, j, LA.norm(2 * la_old * Fy - la_old * Fy_old) - a *
        # LA.norm(y - y_old)
        x1 = prox(x - la * Fy, la)
        err = LA.norm(x1 - y) + LA.norm(y - x)
        n_F += j + 1
        values.append(err)
        res = [values, x1, x, y, la, Fy, tau, n_F, err]
        return res

    for i in xrange(numb_iter):
        iterates = T(*iterates)
        err = iterates[-1]
        if err <= tol:
            end = time()
            n_F = iterates[-2]
            print "---- Alg. 1 ----"
            print "Number of iterations:", i + 1
            print "Number of prox_g:", i + 1
            print "Number of F, :", n_F
            print "Time execution:", np.round(end - begin, 2)
            break
    if err > tol:
        print "alg_VI_proj does not terminate after", numb_iter, "iterations"

    return [iterates[i] for i in [0, 1]]


def alg_VI_proj_affine(F, prox, x0, tau_0=1, constr='True',  numb_iter=100, tol=1e-6):
    """
    Implementation of the Algorithm 1 from the paper in case when F is affine. 
    The docs are the same as for the function alg_VI_proj.
    """
    begin = time()
    a = 0.41
    sigma = 0.7
    la_max = 1000
    values = []
    init = initial_lambda(F, x0, a)
    iterates = [values] + init + [init[-1]] + [tau_0, 2, 1]

    def T(values, x, x_old, y_old, la_old, Fx_old, Fy_old, tau_old, n_F, err):
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
        err = LA.norm(x1 - y) + LA.norm(y - x)
        values.append(err)
        n_F += 1
        res = [values, x1, x, y, la, Fx, Fy, tau, n_F, err]
        return res

    for i in xrange(numb_iter):
        iterates = T(*iterates)
        err = iterates[-1]
        if err <= tol:
            end = time()
            n_F = iterates[-2]
            print "---- Alg. 1 ----"
            print "Number of iterations:", i + 1
            print "Number of prox_g:", i + 1
            print "Number of F, :", n_F
            print "Time execution:", np.round(end - begin, 2)
            break
    if err > tol:
        print "alg_VI_proj_affine does not terminate after", numb_iter, "iterations"

    return [iterates[i] for i in [0, 1]]


def alg_VI_prox_minim(F, prox, x0, tau0=1, theta=2, numb_iter=100, tol=1e-6):
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
    numb_iter: positive integer, optional. Number of iteration. 
    tol: stopping tolerance, the algortithm terminates when
    ||x1-y||+||y-x|| <= tol

    Returns:
    list of 2 elements: [values, x1]. Values collects the history of
    all err during all iterations.
    """
    begin = time()
    tau = tau0
    a = 0.41 * (2 - 1. / theta)
    sigma = 0.7
    values = []
    iterates = [values] + initial_lambda(F, x0, a) + [tau, 2, 1]

    def T(values, x, x_old, y_old, la_old, Fy_old, tau_old, n_F, err):
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
        err = LA.norm(x1 - y) + LA.norm(y - x)
        values.append(err)
        res = [values, x1, x, y, la, Fy, tau, n_F, err]
        return res

    for i in xrange(numb_iter):
        iterates = T(*iterates)
        err = iterates[-1]
        if err <= tol:
            end = time()
            n_F = iterates[-2]
            print "---- Alg. 3 ----"
            print "Number of iterations:", i + 1
            print "Number of prox_g:", i + 1
            print "Number of F, :", n_F
            print "Time execution:", np.round(end - begin, 2)
            break
    if err > tol:
        print "alg_VI_prox_minim does not terminate after", numb_iter, "iterations"

    return [iterates[i] for i in [0, 1]]


#########

def tseng_fbf_linesearch(F, prox_g, x0, delta=2, numb_iter=100, tol=1e-6):
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
    values = []
    iterates = [values, x0, la0, 1, 0, 1]

    def T(values, x, la, n_F, n_prox, err):
        Fx = F(x)
        la *= delta
        for j in count(0):
            z = prox_g(x - la * Fx, la)
            err = LA.norm(z - x)
            Fz = F(z)
            if la * LA.norm(Fz - Fx) <= theta * LA.norm(z - x):
                break
            else:
                la *= beta

        x1 = z - la * (Fz - Fx)
        values.append(err)
        n_F = n_F + j + 2 if err > tol else n_F
        n_prox = n_prox + j + 1 if err > tol else n_prox
        ans = [values, x1, la,  n_F, n_prox, err]
        return ans

    for i in xrange(numb_iter):
        iterates = T(*iterates)
        err = iterates[-1]
        if err <= tol:
            end = time()
            n_F = iterates[-3]
            n_prox = iterates[-2]
            print "---- FBF alg.----"
            print "Number of iterations:", i + 1
            print "Number of prox_g:", n_prox
            print "Number of F, :", n_F
            print "Time execution:", np.round(end - begin, 2)
            break
    if err > tol:
        print "tseng_fbf_linesearch does not terminate after", numb_iter, "iterations"

    return [iterates[i] for i in [0, 1]]


def fista_linesearch(f, d_f, prox_g, x0, la0=1, numb_iter=100, tol=1e-6):
    """
    Minimize function F(x) = f(x) + g(x) by FISTA with backtracking.
    Takes J as some evaluation function for comparison.
    """
    begin = time()
    values = []
    iterates = [values, x0, x0, 1, la0, 0, 0, 0, 1]

    def iter_T(values, x, y, t, la, n_f, n_df, n_prox, err):
        dfy = d_f(y)
        fy = f(y)
        sigma = 0.7
        for j in count(0):
            x1 = prox_g(y - la * dfy, la)
            if f(x1) <= fy + np.vdot(dfy, x1 - y) + 0.5 / la * LA.norm(x1 - y)**2:  # F(x1) < Q(x1, y)
                break
            else:
                la *= sigma
        t1 = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
        y1 = x1 + (t - 1) / t1 * (x1 - x)
        n_f += j + 2
        n_df += j + 1
        n_prox += j + 1
        err = LA.norm(x1 - y)
        values.append(err)
        ans = [values, x1, y1, t1, la, n_f, n_df, n_prox, err]
        return ans

    for i in xrange(numb_iter):
        iterates = iter_T(*iterates)
        err = iterates[-1]
        if err <= tol:
            end = time()
            n_f, n_df, n_prox = iterates[-4:-1]
            print "---- FISTA ----"
            print "Number of iterations:", i + 1
            print "Number of prox_g:", n_prox
            print "Number of grad, :", n_df
            print "Number of f, :", n_f
            print "Time execution:", np.round(end - begin, 2)
            break
    if err > tol:
        print "FISTA does not terminate after", numb_iter, "iterations"

    return [iterates[i] for i in [0, 1]]


def prox_grad_linesearch(f, d_f, prox_g, x0, la0=1, numb_iter=100, tol=1e-6):
    """
    Minimize function F(x) = f(x) + g(x) by proximal gradient method
    with backtracking.  
    """
    begin = time()
    values = []
    fx0 = f(x0)
    iterates = [values, x0, fx0, la0, 0, 0, 0, 1]

    def iter_T(values, x, fx, la, n_f, n_df, n_prox, err):
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
        err = LA.norm(x1 - x)
        # notice that I increase stepsize
        ans = [values, x1, fx1,  1.5 * la, n_f, n_df, n_prox, err]
        return ans

    for i in xrange(numb_iter):
        iterates = iter_T(*iterates)
        err = iterates[-1]
        if err <= tol:
            end = time()
            n_f, n_df, n_prox = iterates[-4:-1]
            print "---- PGM ----"
            print "Number of iterations:", i + 1
            print "Number of prox_g:", n_prox
            print "Number of function, n_f:", n_f
            print "Number of gradients, n_grad:", n_df

            print "Time execution:", end - begin

    if err > tol:
        print "Proximal gradient method does not terminate after", numb_iter, "iterations"

    return [iterates[i] for i in [0, 1]]


def fb_cruz_nghia1(df, prox_g, x0, a0, beta=1.5, numb_iter=100, tol=1e-6):
    """
    Minimize function F(x) = f(x) + g(x) by proximal gradient method
    with linesearch, proposed in [1] (Method 1)


    [1] J.Cruz, T.Nghia 'On the convergence of the forward-backward splitting method with
    linesearches'

    """

    begin = time()
    th = 0.7
    delta = 0.49
    values = []
    iterates = [values, x0, df(x0), a0, 0, 0, 1]

    def T(values, x, df_x, a, n_prox, n_df, err):
        for i in count(0):
            z = prox_g(x - a * df_x, a)
            df_z = df(z)
            if a * LA.norm(df_z - df_x) <= delta * LA.norm(z - x):
                break
            else:
                a *= th
        # print a
        n_prox += i + 1
        n_df += i + 1
        err = LA.norm(z - x)
        values.append(err)
        res = [values, z, df_z, beta * a, n_prox, n_df, err]
        return res

    for i in range(numb_iter):
        iterates = T(*iterates)
        err = iterates[-1]
        if err <= tol:
            n_prox = iterates[-3]
            n_df = iterates[-2]
            end = time()
            print "---- Forward-backward method-1 ----"
            print "Number of iterations:", i + 1
            print "Number of prox_g:", n_prox
            print "Number of df, :", n_df
            print "Time execution:", np.round(end - begin, 2)
            break
    if err > tol:
        print "Forward-backward method-1  does not terminate after", numb_iter, "iterations"

    return [iterates[i] for i in [0, 1]]


def fb_cruz_nghia3(f, g, df, prox_g, x0, numb_iter=100, tol=1e-6):
    begin = time()
    th = 0.7
    delta = 0.49
    values = []
    iterates = [values, x0, 0, 0, 1]

    def T(values, x, n_prox, n_f, err):
        df_x = df(x)
        z = prox_g(x - df_x, 1)
        beta = 1
        for i in count(0):
            x1 = x - beta * (x - z)
            if f(x1) + g(x1) <= f(x) + g(x) - beta * (g(x) - g(z)) - beta * df_x.dot(x - z) + beta / 2 * LA.norm(x - z)**2:
                break
            else:
                beta *= th
        n_prox += i + 1
        n_f += i + 1
        err = LA.norm(z - x)
        x = x1
        values.append(err)
        res = [values, x, n_prox, n_f, err]
        return res

    for i in range(numb_iter):
        iterates = T(*iterates)
        err = iterates[-1]
        if err <= tol:
            n_prox = iterates[-3]
            n_df = iterates[-2]
            end = time()
            print "---- Forward-backward method-3 ----"
            print "Number of iterations:", i + 1
            print "Number of prox_g:", n_prox
            print "Number of df, :", n_df
            print "Time execution:", np.round(end - begin, 2)
            break
    if err > tol:
        print "Forward-backward method-3  does not terminate after", numb_iter, "iterations"

    return iterates
