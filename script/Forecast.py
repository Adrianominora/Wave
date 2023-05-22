from lmfit import Model, Parameters
import numpy as np

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
    params.add("K", 10, min=0, max=100)
    params.add("b", 1, min=1e-2, max=100)
    params.add("c", 1, min=1e-6, max=2)
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

def compute_start(new_pos, window_size=7, start_0=14):
    tt=np.arange(0,new_pos.shape[0])
    rt = np.zeros(new_pos.shape)
    new_pos_fit=np.zeros(new_pos.shape)
    for t in range(window_size,tt[-1]):
        coefficients = np.polyfit(tt[t-window_size:t], new_pos[t-window_size:t], 0)
        new_pos_fit[t] = np.polyval(coefficients, t)

    for t in range(window_size,tt[-1]):
        rt[t] = compute_rt(new_pos_fit[0:t],window_size)

    t_start = [start_0]
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
    norm_factor = 1
    params = make_param_0()
    # def __init__(self) :
        # if params == None:
        #     self.params = Parameters()
        #     self.params = make_param_0()
        # else:
        #     self.params = Parameters()
        #     self.params = params
        

    # def make_param_0(self):
    #     self.params.add(f"K", 10, min=0, max=100)
    #     self.params.add(f"b", 1, min=1e-2, max=100)
    #     self.params.add(f"c", 1, min=1e-6, max=2)
    #     #params.add(f"d", 0, min=0)

    # def set_norm_factor(self, nf):
    #     self.norm_factor(nf)
    
    def predict(self, tt):
        prediction = self.norm_factor*gompertz(tt, self.params['K'].value, self.params['b'].value, self.params['c'].value)
        return prediction
        
    def fit(self, new_pos, max_nfev = 1000) :
        self.norm_factor = np.amax(new_pos)
        t0=0
        T_tot = new_pos.shape[0]
        if not(T_tot>t0):
            print('Final time has to be greater than initial time')
            raise
        new_pos_avg = rolling_avg(new_pos)
        # Create model
        fmodel= Model(gompertz)
        # fit the model
        result = fmodel.fit(new_pos/self.norm_factor, self.params, 
                            t=np.arange(t0,T_tot), max_nfev=max_nfev)
        # Update
        self.params = result.params

    def set_params(self, params):
        self.params=params

# Multiple waves fit
class n_waves:
    def __init__(self, new_pos_tot):
        self.new_pos_tot=new_pos_tot
        self.L_waves = [wave_fit()]
        self.n = 1
        self.overlap = 14
        self.L_starts = [self.overlap]

    def fit(self, T):
        for t in range (self.overlap,T):
            t_start = compute_start(self.new_pos_tot[0:t], window_size=7, start_0=self.overlap)
            t_start_new = t_start[-1]

            if self.L_starts[-1] != t_start_new:
                WF = wave_fit()
                self.L_waves.append(WF)
                self.L_starts.append(t_start_new)
                self.n += 1

            if(self.n==1):
                self.L_waves[-1].fit(self.new_pos_tot[0:t])
            else:
                new_pos_t = self.new_pos_tot[self.L_starts[-1] - self.overlap  : t]
                tail_pos = self.L_waves[-1].predict(np.arange(self.L_starts[-1] - self.L_starts[-2], t - (self.L_starts[-2] - self.overlap)))
                self.L_waves[-1].fit(new_pos_t-tail_pos)

    def single_predict(self, wave_index, tt):
        """This return a single wave contribution in a given interval expressed as global time"""
        if wave_index >= self.n:
            print('Invalid wave index')
            raise
        wave = self.L_waves[wave_index]
        tt_local = tt - (self.L_starts[wave_index] - self.overlap)
        return wave.predict(tt_local)

    def predict(self, tt):
        """This return the current prediction considering all waves contributions"""
        sum = np.zeros(tt.shape)
        for i in range(self.n):
            sum += self.single_predict(i, tt)
        return sum

    # forecast_list+=WF.forecast_list[0]
    # corrected_forecast_list += WF.corrected_forecast_list[0]