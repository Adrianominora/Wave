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

def compute_rt(func, window_size=14):
    data = func[-(window_size+1):-1]
    log_data = np.log(data)
    rt = np.zeros((log_data.shape[0]-1,))
    for i in range(rt.shape[0]):
        rt[i] = log_data[i+1]-log_data[i]
    return np.mean(rt)

def rolling_avg(new_pos, window_size=7):
    new_pos_avg=np.empty(new_pos.shape)
    for i in range(window_size):
        weight = np.arange(1,i+2)
        weight=weight/np.linalg.norm(weight)
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

def copute_conc(func, window_size=14):
    norm_factor=np.amax(func)
    data = func[-(window_size+1):-1]/norm_factor
    dif = np.zeros((data.shape[0]-1,))
    concv = np.zeros((dif.shape[0]-1,))
    for i in range(dif.shape[0]):
        dif[i] = data[i+1]-data[i]
    for i in range(concv.shape[0]):
        concv[i] = dif[i+1]-dif[i]
    return np.mean(concv)

def compute_start(new_pos, window_size=7):
    tt=np.arange(0,new_pos.shape[0])
    rt = np.zeros(new_pos.shape)
    new_pos_fit=np.zeros(new_pos.shape)
    for t in range(window_size,tt[-1]):
        coefficients = np.polyfit(tt[t-window_size:t], new_pos[t-window_size:t], 0)
        new_pos_fit[t] = np.polyval(coefficients, t)

    for t in range(window_size,tt[-1]):
        rt[t] = compute_rt(new_pos_fit[0:t],window_size)

    t_start = [0]
    t_end = [50]
    t=0
    while t<tt[-1]:
        if rt[t]>0.04 and t>t_end[-1] and t>t_start[-1]+40 and len(t_end)==len(t_start):
            t_start.append(t)
        if rt[t]<-0.025 and t>t_start[-1]+20 and t>t_end[-1]+40 and len(t_end)==len(t_start)-1:
            t_end.append(t)
        t+=1

    return t_start

# Single wave fit
class wave_fit:
    def __init__(self, new_pos_tot) :
        self.param_list=[]
        self.forecast_list=[]
        self.corrected_forecast_list=[]
        self.conc_list=[]
        self.fit_list=[]
        self.new_pos_tot=new_pos_tot
        self.new_pos_avg = 0
        self.norm_factor = np.amax(new_pos_tot)

    def make_forecast(new_pos, params, T = 14,max_nfev=10000):
        t0 = 0
        today = new_pos.shape[0]
        cum_pos = np.cumsum(new_pos)
        norm_factor = np.amax(new_pos)
        data = new_pos/norm_factor # Normalization
        cum_data = np.cumsum(data)
        d0 = cum_data[0]
        tt=np.arange(t0,today)
        tt_forecast=np.arange(t0,today+T)

        # Create model
        fmodel= Model(gompertz)

        # fit the model
        result = fmodel.fit(data, params, t=tt-t0, max_nfev=max_nfev)
        sigma = result.eval_uncertainty()
        fit = norm_factor*gompertz(tt_forecast-t0+3.5, result.params['K'].value, result.params['b'].value, result.params['c'].value)
        forecast = fit[today+1]
        return fit, result.params, forecast
    
    def predict(self, tt):
        prediction = self.norm_factor*gompertz(tt, self.param_list[-1]['K'].value, self.param_list[-1]['b'].value, self.param_list[-1]['c'].value)

        return prediction
        
    def fit(self,t0,T_tot):
        if not(T_tot>t0):
            print('Final time has to be greater than initial time')
            raise
        for t in range(max(t0,14),T_tot):
            new_pos = self.new_pos_tot[t0-14:t]  # Data acquisition
            self.new_pos_avg = rolling_avg(new_pos)

            daily_err = compute_derr(new_pos,self.new_pos_avg)
            # daily_err = np.load('../data/daily_errors.npy')[(t+1-3)%7]

            # Compute concavity
            self.conc_list.append(copute_conc(self.new_pos_avg))

            # Correct correction
            if t>t0+24:
                daily_err += 75*self.conc_list[-1]
            
            # Define parameters
            if len(param_list)==0:
                params = make_param_0()
            else:
                params = param_list[-1]
            
            fit, result_params, forecast= make_forecast(self.new_pos_avg, params)
            
            # Saving parameters and forecast
            self.param_list.append(result_params)
            self.forecast_list.append(forecast)
            self.corrected_forecast_list.append(forecast*(1+daily_err))
            self.fit_list.append(fit)

        def set_params(params):
            self.param_list=[params]
        
        @ property
        def param_list(self):
            return self.param_list
        
        @ property
        def forecast_list(self):
            return self.forecast_list
        
        @ property
        def corrected_forecast_list(self):
            return self.corrected_forecast_list
        
        @ property
        def conc_list(self):
            return self.conc_list
        
        @ property
        def fit_list(self):
            return self.fit_list
        
        @ property
        def new_pos_avg(self):
            return self.new_pos_avg
        
        @ property
        def norm_factor(self):
            return self.norm_factor
        
    class n_waves:
        def __init__(self, new_pos_tot):
            self.new_pos_tot=new_pos_tot
            self.L_waves = [0]
            self.L_starts = [0]
            self.n = 1

        def fit(self, T):

            t0=0
            for t in range (14,T):
                t_start = compute_start(self.new_pos_tot[0:t], window_size=7)
                t_start_new = t_start[-1]


                if(self.n==1):
                    if self.L_starts[-1] != t_start_new:
                        self.L_starts.append(t_start_new)
                        self.n += 1

                        
                        WF = wave_fit(self.new_pos_tot[0,t])
                        WF.fit(t,t+1)
                        self.L_waves.append(WF)

                    else:
                        WF = wave_fit(self.new_pos_tot[0,t])
                        WF.fit(t,t+1)
                        self.L_waves[self.n-1] = WF
                
                else:
                    if self.L_starts[-1] != t_start_new:
                        self.L_starts.append(t_start_new)
                        self.n += 1
                        
                        new_pos_t = self.new_pos_tot[self.L_starts[-1]  , t]

                        last_params = self.L_waves[-1].param_list[-1]

                        fit = self.L_waves[-1].norm_factor*gompertz(np.arange(0, self.L_starts[-1] - self.L_starts[-2]), params['K'].value, params['b'].value, params['c'].value)
                        forecast = fit[today+1]
                        diff_pos = new_pos_t - make_forecast(new_pos_t,self.L_waves[-1].param_list[-1], t )


                        WF = wave_fit(self.new_pos_tot[self.L_starts[-1],t] - )
                        WF.fit(t,t+1)
                        self.L_waves.append(WF)

                    else:
                        WF = wave_fit(self.new_pos_tot[self.L_starts[-1]-14,t])
                        WF.fit(t,t+1)
                        self.L_waves[self.n-1] = WF


                forecast_list+=WF.forecast_list[0]
                corrected_forecast_list = WF.corrected_forecast_list