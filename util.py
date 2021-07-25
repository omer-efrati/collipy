import numpy as np
from scipy import stats, odr


def fit_ols(x, dx, y, dy, func, beta_guess):
    """Fitting a Function Using OLS hence assuming dx << dy"""
    model = odr.Model(func)
    data = odr.RealData(x=x, sx=dx, y=y, sy=dy)
    myodr = odr.ODR(data, model, beta0=beta_guess)
    myodr.set_job(fit_type=2)
    output = myodr.run()
    chi, p_value = gof(x, dx, y, dy, func, output.beta)
    return output, chi, p_value


def gof(x, dx, y, dy, func, beta):
    def f(x): return func(beta, x)

    res = y - f(x)
    df = len(x) - len(beta)
    chiseq = res.T @ np.linalg.inv(np.diag(dy ** 2 + ((f(x + dx) - f(x - dx)) / 2) ** 2)) @ res
    p_value = 1 - stats.chi2.cdf(chiseq, df)
    return chiseq / df, p_value
