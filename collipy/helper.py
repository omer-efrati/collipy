"""
Helper module for collipy package
"""
import numpy as np
from scipy import stats, odr


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
