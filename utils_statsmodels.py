from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.regression.linear_model import RegressionResultsWrapper
from pandas import Series, DataFrame


def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
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

def dynamic_pred(model: RegressionResultsWrapper, exog: DataFrame, lag_endog: str) -> DataFrame:
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
        ftbl['mean_ci_upper'].append(tbl['mean_ci_lower'].values[0])
        ftbl['obs_ci_lower'].append(tbl['obs_ci_lower'].values[0])
        ftbl['obs_ci_upper'].append(tbl['obs_ci_upper'].values[0])

        if (i+1) > steps-1:
            break
                
        exog.loc[ids[i+1], lag_endog] = tbl['mean'].values[0]

    df_ftbl = DataFrame(ftbl).set_index(exog.index)

    return df_ftbl