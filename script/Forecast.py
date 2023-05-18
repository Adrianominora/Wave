from lmfit import Model, Parameters
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

# Define the Gompertz bell curve function
def gompertz(t, K, b, c):
    return K*b*c*np.exp(-c*(t))*np.exp(-b*np.exp(-c*(t)))

# Fit on the S curve
def cum_gompertz(t, K, b, c):
    return np.cumsum(gompertz(t, K, b, c))

def make_forecast(new_pos, params, max_nfev=10000):
    t0 = 0
    today = new_pos.shape[0]
    cum_pos = np.cumsum(new_pos)
    norm_factor = np.amax(new_pos)
    data = new_pos/norm_factor # Normalization
    cum_data = np.cumsum(data)
    d0 = cum_data[0]
    tt=np.arange(t0,today)
    tt_forecast=np.arange(t0,today+14)

    # Create model
    fmodel= Model(gompertz)

    # fit the model
    result = fmodel.fit(data, params, t=tt-t0, max_nfev=max_nfev)
    sigma = result.eval_uncertainty()
    fit = norm_factor*gompertz(tt_forecast-t0, result.params['K'].value, result.params['b'].value, result.params['c'].value)
    forecast = fit[today+1]
    return fit, result.params, forecast


        
