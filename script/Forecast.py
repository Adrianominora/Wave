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
    params.add("K", 10, min=5, max=100)
    params.add("b", 1, min=1, max=100)
    params.add("t_max", 20, min=15, max=100)
    params.add("c", 1, expr='log(b)/t_max')
    #params.add(f"d", 0, min=0)

    return params

def compute_rt(func, window_size=14):
    data = func[-(window_size):]
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
    tt=np.arange(new_pos[0],new_pos.shape[0])
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
    
    def predict(self, tt):
        prediction = self.norm_factor*gompertz(tt, self.params['K'].value, self.params['b'].value , self.params['c'].value)
        return prediction
        
    def fit(self, new_pos, max_nfev = 1000, window=1):
        self.norm_factor = np.amax(new_pos)
        t0=0
        T_tot = new_pos.shape[0]
        if not(T_tot>t0):
            print('Final time has to be greater than initial time')
            raise
        # Create model
        fmodel= Model(gompertz)
        # fit the model
        if window<1:
            print('Invalid time window for interpreting data')
            raise
        elif window==1:
            result = fmodel.fit(new_pos/self.norm_factor, self.params, 
                                t=np.arange(t0,T_tot), max_nfev=max_nfev)
        else:
            new_pos_avg = rolling_avg(new_pos, window_size=window)
            result = fmodel.fit(new_pos_avg/self.norm_factor, self.params, 
                                t=np.arange(t0,T_tot)-window/2, max_nfev=max_nfev)

        # Update
        self.params = result.params

    def set_params(self, params):
        self.params=params

# Multiple waves fit
class n_waves:
    def __init__(self, new_pos_tot, window=1):
        self.new_pos_tot=new_pos_tot
        self.L_waves = [wave_fit()]
        self.n = 1
        self.overlap = 14
        self.L_starts = [self.overlap]
        self.window = window
        self.t=self.overlap - 1
        self.rt = 0

    def update_tstart(self, poly_w=7): #"""controlla"""
        flag = False
        new_pos_fit = np.zeros(poly_w)
        tt = np.arange(self.t-poly_w, self.t)
        for i in range(poly_w):
            t = tt[i]
            coefficients = np.polyfit(np.arange(t - poly_w, t), self.new_pos_tot[t - poly_w: t], 0)
            new_pos_fit[i] = np.polyval(coefficients, t)
        rt_fit = compute_rt(self.predict(tt))
        self.rt = compute_rt(new_pos_fit, window_size=poly_w) -rt_fit
        if self.rt>0.04 and self.t>self.L_starts[-1]+40 :
            self.L_starts.append(self.t)
            flag = True
        return flag


    def daily_update(self):
        self.t +=1
        t = self.t
        new_wave_flag = self.update_tstart()

        if new_wave_flag:
            WF = wave_fit()
            self.L_waves.append(WF)
            self.n += 1

        if(self.n==1):
            self.L_waves[-1].fit(self.new_pos_tot[0:t], window=self.window)
            return self.new_pos_tot[t]
        else:
            new_pos_t = self.new_pos_tot[self.L_starts[-1] - self.overlap  : t]
            # tail_pos = self.L_waves[-2].predict(np.arange(self.L_starts[-1] - self.L_starts[-2], t - (self.L_starts[-2] - self.overlap)))
            tail_pos = self.predict(np.arange(self.L_starts[-1] - self.overlap  , t), self.n-1 )
            self.L_waves[-1].fit(new_pos_t-tail_pos, window=self.window)
            return (new_pos_t-tail_pos)[-1]
        
    def fit(self, T):
        for t in range (self.overlap,T):
            self.daily_update()

            
    def single_predict(self, wave_index, tt):
        """This return a single wave contribution in a given interval expressed as global time"""
        if wave_index >= self.n:
            print('Invalid wave index')
            raise
        wave = self.L_waves[wave_index]
        tt_local = tt - (self.L_starts[wave_index] - self.overlap)
        return wave.predict(tt_local)

    def predict(self, tt, n=None):
        """This return the current prediction considering all waves contributions"""
        if n==None:
            n=self.n
        sum = np.zeros(tt.shape)
        for i in range(n):
            sum += self.single_predict(i, tt)
        return sum