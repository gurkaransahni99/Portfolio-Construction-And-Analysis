import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import norm

def drawdown(return_series:pd.Series):
    wealth_index=1000*(1+return_series).cumprod()
    prev_peaks=wealth_index.cummax()
    drawdown=(wealth_index-prev_peaks)/prev_peaks
    return pd.DataFrame({
        "Wealth":wealth_index,
        "Peaks":prev_peaks,
        "Drawdown":drawdown
    })

def get_ffme_returns():
    me_m = pd.read_csv("Portfolios_Formed_on_ME_monthly_EW.csv", header=0, index_col=0, parse_dates=True, na_values=-99.99)
    rets=me_m[['Lo 10', 'Hi 10']]
    rets.columns=["SmallCap", "LargeCap"]
    rets=rets/100
#     rets.plot.line()
    rets.index=pd.to_datetime(rets.index, format="%Y%m")
    return rets

def get_hfi_returns():
    hfi=pd.read_csv("edhec-hedgefundindices.csv", header=0, index_col=0, parse_dates=True)
    hfi=hfi/100
    hfi.index=hfi.index.to_period("M")
    return hfi

def get_ind_returns():
    ind = pd.read_csv("ind49_m_vw_rets.csv", header=0, index_col=0, parse_dates=True)/100
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period("M")
    ind.columns=ind.columns.str.strip()
    return ind

def semideviation(r):
    return r[r<0].std(ddof=0)

def skewness(r):
    demeaned_r=r-r.mean()
    sigma_r=r.std(ddof=0)
    exp=(demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    demeaned_r=r-r.mean()
    sigma_r=r.std(ddof=0)
    exp=(demeaned_r**4).mean()
    return exp/sigma_r**4

def is_normal(r, level=0.01):
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value>level
    
def var_historic(r, level=5):
    if(isinstance(r, pd.DataFrame)):
        return r.aggregate(var_historic, level=level)
    elif(isinstance(r, pd.Series)):
        return -np.percentile(r,level)
    else:
        raise TypeError("Expected series or dataframe")
        
def var_gaussian(r, level=5, modified = False):
    z=norm.ppf(level/100)
    
    if(modified):
        s=skewness(r)
        k=kurtosis(r);
        z=(z+
              (z**2-1)*s/6+
              (z**3-3*z)*(k-3)/24-
              (2*z**3-5*z)*(s**2)/36
          )
    
    return -(r.mean()+z*r.std(ddof=0))

def cvar_historic(r, level=5):
    if(isinstance(r, pd.DataFrame)):
        return r.aggregate(cvar_historic, level=level)
    elif(isinstance(r, pd.Series)):
        val_to_calc=r<=-(var_historic(r, level=level))
        return -(r[val_to_calc].mean())
    else:
        raise TypeError("Expected series or dataframe")

        
