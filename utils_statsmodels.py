from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.regression.linear_model import RegressionResultsWrapper
from pandas import Series, DataFrame


def adf_test(series, terms='c', title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC', regression=terms) # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val

    out['Terms'] = terms
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Data has a unit root and is non-stationary")

    
def bglm_test(model, lags, title='') -> Series:
    """
    Pass in a fitted model and return the Breusch-Godfrey LM Test
    """
    print(f'Breusch-Godfrey LM Test: {title}')
    result = acorr_breusch_godfrey(model, lags)
    
    labels = ['LM test statistic','LM (Chi-sq) p-value', 'F test statistic','F p-value']
    out = Series(result,index=labels)
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"

    print(f'Null hypothesis: No Autocorrelation of any order up to {lags} lags.')

    if result[1] <= 0.05:
        print("Reject the null hypothesis at the .05 significance level")
        print("Evidence of Autocorrelation in the residuals.")
    else:
        print("Fail to reject the null hypothesis at the .05 significance level")
        print("Data has no evidence of autocorrelation.")

def dynamic_pred(
    model: RegressionResultsWrapper, 
    exog: DataFrame, 
    lag_endog: str) -> DataFrame:
    """
    Takes a fitted model, a set of exogenous variables, and the name of the lagged
    endogenous variable and returns a DataFrame with the summary of the dynamic prediction.
    """
    exog = exog.copy()
    steps = exog.shape[0]
    ids = exog.index 
    fcast = None
    ftbl = {'mean': [], 'mean_se': [], 
            'mean_ci_lower': [], 'mean_ci_upper': [], 
            'obs_ci_lower': [], 'obs_ci_upper': []}

    #predictions = model.get_prediction(exog.iloc[[0]]).summary_frame()
    #return predictions

    for i, step in enumerate(range(steps)):
        predictions = model.get_prediction(exog.iloc[[i]])
        tbl = predictions.summary_frame()

        ftbl['mean'].append(tbl['mean'].values[0])
        ftbl['mean_se'].append(tbl['mean_se'].values[0])
        ftbl['mean_ci_lower'].append(tbl['mean_ci_lower'].values[0])
        ftbl['mean_ci_upper'].append(tbl['mean_ci_upper'].values[0])
        ftbl['obs_ci_lower'].append(tbl['obs_ci_lower'].values[0])
        ftbl['obs_ci_upper'].append(tbl['obs_ci_upper'].values[0])

        if (i+1) > steps-1:
            break
                
        exog.loc[ids[i+1], lag_endog] = tbl['mean'].values[0]

    df_ftbl = DataFrame(ftbl).set_index(exog.index)

    return df_ftbl

from numpy import zeros, dot, quantile, mean
from numpy.random import choice
from numpy.linalg import lstsq
from statsmodels.api import OLS

def stochastic_forecast(
    model: RegressionResultsWrapper, 
    X_train: DataFrame, 
    X_test: DataFrame,
    ci: float = .95, 
    simulations: int = 1000) -> DataFrame:
    """
    Use bootstrapped residuals to perform stochastic forecasting.

    params:
    model: a fitted OLS object from `statsmodels`
    X_train: the exogenous variables for the training set.
    X_test: the exogenous variables for the forecasting period.
    interval: the confidence interval.
    simulations: the number of bootstrapped repetitions.
    """
    
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    test_size = X_test.shape[0]

    baseline = model.predict()
    len_baseline = len(baseline)
    residuals = model.resid
    
    X_train = X_train.rename(columns={"const": "Intercept"})
    X_test = X_test.rename(columns={"const": "Intercept"})

    # Boot
    bootstraps = zeros((simulations, test_size))
    for i in range(simulations):
        # boot_params = [np.random.normal(mean, sd) for mean, sd in zip(params, params_se)]

        boot_residuals = choice(residuals, len_baseline, replace=True)
        boot_y = baseline + boot_residuals

        params, *_ = lstsq(X_train, boot_y, rcond=None)

        pred = dot(X_test, params)
        
        bootstraps[i, :] = pred

    final_frame = zeros((test_size, 3))
    
    quantile_intervals = [.5 - (ci/2), .5, .5 + (ci/2)]

    for i in range(test_size):
        final_frame[i, :] = quantile(bootstraps[:, i], quantile_intervals)

    final_frame = DataFrame(final_frame).set_index(X_test.index)
    final_frame.columns = ['lower_q', 'median', 'upper_q']

    return final_frame


def stochastic_forecast2(
    model, 
    X_train, 
    X_test,
    ci=0.95,
    simulations = 1000):
        
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    test_size = X_test.shape[0]

    baseline = model.predict()
    len_baseline = len(baseline)
    residuals = model.resid
    
    X_train = X_train.rename(columns={"const": "Intercept"})
    X_test = X_test.rename(columns={"const": "Intercept"})

    X_train = X_train.assign(Intercept=1.)
    X_test = X_test.assign(Intercept=1.)

    # Boot
    boot_mean = zeros((simulations, test_size))
    boot_upper = zeros((simulations, test_size))
    boot_lower = zeros((simulations, test_size))
    for i in range(simulations):
        # boot_params = [np.random.normal(mean, sd) for mean, sd in zip(params, params_se)]

        boot_residuals = choice(residuals, len_baseline, replace=True)
        boot_y = baseline + boot_residuals

        pred = OLS(boot_y, X_train).fit().get_prediction(X_test).summary_frame(alpha=ci)

        boot_mean[i, :] = pred['mean']
        boot_upper[i, :] = pred['obs_ci_upper']
        boot_lower[i, :] = pred['obs_ci_lower']

    final_frame = zeros((test_size, 3))
    
    for i in range(test_size):
        final_frame[i, :] = [mean(boot_mean[:, i]), mean(boot_lower[:, i]), mean(boot_upper[:, i])]

    final_frame = DataFrame(final_frame).set_index(X_test.index)
    final_frame.columns = ['mean', 'lower', 'upper']

    return final_frame

from numpy import log

def aic_eviews(model):
    df = model.df_model
    llf = model.llf
    n = model.nobs
    return -2 * (llf/n) + 2 * (df/n)

def bic_eviews(model):
    df = model.df_model
    llf = model.llf
    n = model.nobs
    return -2 * (llf/n) + df * log(n) / (n)

def aic_lectures(model):
    df = model.df_model
    sse = model.sse
    n = model.nobs
    return n * log(sse) + 2 * df

def bic_lectures(model):
    df = model.df_model
    sse = model.sse
    n = model.nobs
    return n * log(sse) + df * log(n)