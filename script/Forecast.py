from lmfit import Model, Parameters
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

# Sinusoidal function
def sin_wave(t, A, T, c):
    return c + A*np.sin(2*np.pi*t/T)

# Define the Gompertz bell curve function
def gompertz(t, K, b, c):
    return K*b*c*np.exp(-c*(t))*np.exp(-b*np.exp(-c*(t)))

# Fit on the S curve
def cum_gompertz(t, K, b, c):
    return np.cumsum(gompertz(t, K, b, c))


def make_param_0():
    params = Parameters()
    params.add(f"K", 10, min=0, max=100)
    params.add(f"b", 1, min=1e-2, max=100)
    params.add(f"c", 1, min=1e-6, max=2)
    #params.add(f"d", 0, min=0)

    return params

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
    result = fmodel.fit(data, params, t=tt-t0-7, max_nfev=max_nfev)
    sigma = result.eval_uncertainty()
    fit = norm_factor*gompertz(tt_forecast-t0, result.params['K'].value, result.params['b'].value, result.params['c'].value)
    forecast = fit[today+1]
    return fit, result.params, forecast

def compute_rt(new_pos, window_size=14):
    data = new_pos[-(window_size+1):-1]
    log_data = np.log(data)
    rt = np.zeros((log_data.shape[0]-1,))
    for i in range(rt.shape[0]):
        rt[i] = log_data[i+1]-log_data[i]
    return np.mean(rt)

def rolling_avg(new_pos, window_size=7):
    new_pos_avg=np.empty(new_pos.shape)
    for i in range(window_size):
        weight = np.arange(1,i+1)
        new_pos_avg[i]=np.mean(new_pos[:i+1])

    weight = np.arange(1,window_size+1)
    weight=weight/np.linalg.norm(weight)
    for i in range(window_size,new_pos_avg.shape[0]):
        new_pos_avg[i]=np.mean(new_pos[i-(window_size-1):i+1])
    return new_pos_avg

def compute_derr(new_pos, new_pos_avg, shift_day=1):
    avg=new_pos_avg[3:]
    pos=new_pos[3:]
    err=(pos-avg)/avg
    res=err.shape[0]%7
    n_weeks=int(err.shape[0]/7)
    err=err[res:]
    err=np.reshape(err,(7,n_weeks), order='F')
    # Extract error of a given day

    i = shift_day
    data=err[i,:]
    fmodel = Model(sin_wave)
    if data.shape[0]<0:
        ttt = np.arange(0, data.shape[0])
        params = Parameters()
        params.add(f"A", 10, min=-2, max=2)
        params.add(f"T", 1, min=data.shape[0]/2, max=5*data.shape[0])
        params.add(f"c", 1, min=-1, max=1)
        result = fmodel.fit(data, params, t=ttt, max_nfev=1000)
        err_pred = sin_wave(ttt[-1] + shift_day, result.params['A'].value, result.params['T'].value, result.params['c'].value)
    else:
        err_pred = np.mean(data)
    return err_pred

def copute_conc(new_pos, new_pos_avg, window_size=14):
    data = new_pos[-(window_size+1):-1]
    dif = np.zeros((data.shape[0]-1,))
    concv = np.zeros((dif.shape[0]-1,))
    for i in range(dif.shape[0]):
        dif[i] = data[i+1]-data[i]
    for i in range(concv.shape[0]):
        concv[i] = dif[i+1]-dif[i]
    return np.mean(concv)

