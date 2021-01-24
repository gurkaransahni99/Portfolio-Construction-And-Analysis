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
    me_m = pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv", header=0, index_col=0, parse_dates=True, na_values=-99.99)
    rets=me_m[['Lo 10', 'Hi 10']]
    rets.columns=["SmallCap", "LargeCap"]
    rets=rets/100
#     rets.plot.line()
    rets.index=pd.to_datetime(rets.index, format="%Y%m")
    return rets

def get_hfi_returns():
    hfi=pd.read_csv("data/edhec-hedgefundindices.csv", header=0, index_col=0, parse_dates=True)
    hfi=hfi/100
    hfi.index=hfi.index.to_period("M")
    return hfi

def get_ind_returns():
    ind = pd.read_csv("data/ind49_m_vw_rets.csv", header=0, index_col=0, parse_dates=True)/100
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

def annualize_returns(r, periods_per_year):
    compounded_growth = (1+r).prod()
    return compounded_growth**(periods_per_year/r.shape[0])-1;
        
def annualize_vol(r, periods_per_year):
    return r.std()*(periods_per_year**0.5)


def sharpe_ratio(r, risk_free_rate, periods_per_year):
    risk_free_rate = (1+risk_free_rate)**(1/periods_per_year)-1
    excess_returns = r - risk_free_rate
    return annualize_returns(excess_returns, periods_per_year)/annualize_vol(r, periods_per_year);

def portfolio_returns(weights, returns):
    return weights.T @ returns

def portfolio_vol(weights, covMat):
    return (weights.T @ covMat @ weights)**0.5

def plot_ef2(n_points, er, cov):
    weights = [np.array([w,1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_returns(w, er) for w in weights]
    volatility = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": volatility
    })
    return ef.plot.scatter(x = "Volatility", y = "Returns", style = "--")

from scipy.optimize import minimize
def minimize_vol(target_return, er, cov):
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds =((0.0, 1.0),)*n
    return_is_target = {
        "type" : "eq",
        "args" : (er,),
        "fun" : lambda weights, er : target_return - portfolio_returns(weights, er)
    }
    
    weights_sum_is_1 = {
        "type" : "eq", 
        "fun" : lambda weights : np.sum(weights) - 1
    }
    
    results = minimize(portfolio_vol,
                      init_guess,
                      args = (cov,),
                      method = "SLSQP",
                      options = {"disp" : False},
                      constraints = (return_is_target, weights_sum_is_1),
                      bounds = bounds
                      )
    
    return results.x
    
def optimal_weights(n_points, er, cov):
    target_ret = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_ret]
    return weights

def plot_ef(n_points, er, cov, show_cml=False, show_ew = False, show_gmv = False, risk_free_rate = 0):
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_returns(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets,
        "Volatility": vols
    })
    ax = ef.plot.scatter(x = "Volatility", y = "Returns", style = "._")
    if(show_cml):
        ax.set_xlim(left = 0)
        w_msr = msr(risk_free_rate, er, cov)
        r_msr = portfolio_returns(w_msr, er)
        v_msr = portfolio_vol(w_msr, cov)
        cml_x = [0, v_msr]
        cml_y = [risk_free_rate, r_msr]
        ax.plot(cml_x, cml_y, color = "green", linestyle = "dashed")
        
    if(show_ew):
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_returns(w_ew, er)
        v_ew = portfolio_vol(w_ew, cov)
        
        ax.plot([v_ew], [r_ew], color = "goldenrod", marker = "o", markersize = 7)
        
    if(show_gmv):
        n = er.shape[0]
        w_ew = msr(0, np.repeat(1, n), cov)
        r_ew = portfolio_returns(w_ew, er)
        v_ew = portfolio_vol(w_ew, cov)
        
        ax.plot([v_ew], [r_ew], color = "midnightblue", marker = "o", markersize = 7)
        
        
def msr(risk_free_rate, er, cov):
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds =((0.0, 1.0),)*n
    weights_sum_is_1 = {
        "type" : "eq", 
        "fun" : lambda weights : np.sum(weights) - 1
    }
    
    def neg_sharpe_ratio(weights, risk_free_rate, er, cov):
        return - (portfolio_returns(weights, er) - risk_free_rate) / portfolio_vol(weights, cov)
    
    results = minimize(neg_sharpe_ratio,
                      init_guess,
                      args = (risk_free_rate, er, cov, ),
                      method = "SLSQP",
                      options = {"disp" : False},
                      constraints = (weights_sum_is_1),
                      bounds = bounds
                      )
    
    return results.x