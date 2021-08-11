"""
Helper module for collipy package
"""
import time
import numpy as np
from scipy import stats, odr, optimize
from timeit import default_timer as timer
import functools


def fit(x, y, sy, func, beta_initial=None, sx=None) -> tuple[odr.Output, float, float]:
    """
    Fitting a Function Using odr from scipy

    Parameters
    ----------
    x : array_like
        Observed data for the independent variable of the regression
    y : array_like
        Observed data for the dependent variable of the regression
    sy : array_like
        Standard deviations of 'y'
    func : function
        function(beta, x) –> y
    beta_initial: array_like, optional
        initial parameter values
    sx : array_like, optional
        Standard deviations of 'x'

    Returns
    -------
    output : odr.Output
        output of ODR run
    chisq_red : float
        Chi^2_reduced value
    p_value : float
        P-value for the fit

    """
    model = odr.Model(func)
    data = odr.RealData(x=x, sx=sx, y=y, sy=sy)
    if beta_initial:
        myodr = odr.ODR(data, model, beta0=beta_initial)
    else:
        myodr = odr.ODR(data, model)
    myodr.set_job(fit_type=2)
    output = myodr.run()
    chisq_red, p_value = gof(x, y, sy, func, output.beta, sx=sx)
    return output, chisq_red, p_value


def gof(x, y, sy, func, beta, sx=None) -> tuple[float, float]:
    """
    Compute the goodness of fit for the given model

    Parameters
    ----------
    x : array_like
        Observed data for the independent variable of the regression
    y : array_like
        Observed data for the dependent variable of the regression
    sy : array_like
        Standard deviations of y
    func : function
        function(beta, x) –> y
    beta: array_like
        parameter values
    sx : array_like, optional
        Standard deviations of 'x'

    Returns
    -------
    chisq_red : float
        Chi^2_reduced value
    p_value : float
        P-value for the fit
    """
    res = y - func(beta, x)
    # delta degrees of freedom
    ddof = len(x) - len(beta)
    if sx is None:
        chiseq = res.T @ np.linalg.inv(np.diag(sy ** 2)) @ res
    else:
        chiseq = res.T @ np.linalg.inv(np.diag(sy ** 2 + ((func(beta, x + sx) - func(beta, x - sx)) / 2) ** 2)) @ res
    p_value = 1 - stats.chi2.cdf(chiseq, ddof)
    chisq_red = chiseq/ddof
    return chisq_red, p_value


def breit_wigner(beta, x):
    """
                        b_1
    ------------------------------------    +   b_0
    (x**2 - b_2**2)**2 + b_2**2 * b_3**2
    """
    return beta[1] / ((x**2 - beta[2]**2)**2 + beta[2]**2 * beta[3]**2) + beta[0]


def expon(beta, x):
    return beta[0] * np.exp(-x/beta[1]) + beta[2]


def n_sigma(x: tuple[float, float], y: tuple[float, float]) -> float:
    """
    Compute the N_sigma value between `x` and `y`

    Parameters
    ----------
    x
        (value, uncertainty)
    y
        (value, uncertainty)

    Returns
    -------
        N_sigma value

    Notes
    -----
    Fine range of N_sigma value will consider to be under 3.

    """
    nom = np.abs(x[0] - y[0])
    denom = np.sqrt(x[1] ** 2 + y[1] ** 2)
    if nom == 0:
        return 0
    if denom == 0:
        return np.inf
    return nom / denom


def progress_bar(now: int, required: int, start_time=None, note=''):
    fill = f'[{int(30 * now / required) * "#" + (30 - int(30 * now / required)) * "-"}]'
    percent = f'{int(100 * now / required)}% completed'
    dt = (timer() - start_time)
    time = f'{dt/60:0.3g} minutes passed ETC {(required/now - 1) * dt/60:0.3g} minutes' if start_time and now != 0 else ''
    bar = f'{fill} {percent}'
    if time:
        bar += f' {time}'
    if note:
        bar += f' {note}'
    print(f'\r{bar}', end='', flush=True)


def function_timer(function):
    """Print the running time of the decorated function"""
    @functools.wraps(function)
    def function_timer_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = function(*args, **kwargs)
        end = time.perf_counter()
        print(f'Finished `{function.__name__}` run in {end - start: 0.5g}s')
        return result
    return function_timer_wrapper
