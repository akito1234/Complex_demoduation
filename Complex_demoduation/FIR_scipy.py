import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

N = 1024            # サンプル数
dt = 0.001          # サンプリング周期 [s]
f1, f2, f3 = 10, 60, 300 # 周波数 [Hz]

t = np.arange(0, N*dt, dt) # 時間 [s]
x = 3*np.sin(2*np.pi*f1*t) + 0.3*np.sin(2*np.pi*f2*t) + 0.2*np.sin(2*np.pi*f3*t) # 信号

#RRIのリサンプリング
def resamp_PPI(R_x,#心拍の山の時刻
               resamp_frequency=4.0,#リサンプリング周波数
               ):
        
        RRI=np.zeros(R_x.size-1)
        RRI_time=np.zeros(R_x.size-1)
        
        #間隔[ms]を[s]に変換する
        for i in range(R_x.size-1):
            RRI[i]=(R_x[i+1]-R_x[i])*0.001
            RRI_time[i]=R_x[i+1]*0.001


        # 3 次スプライン補間
        # RRI:山の時刻
        # RRI_time : 時間軸
        ff=interpolate.interp1d(RRI_time,RRI,kind='cubic')

        #データ数
        N = resamp_frequency * (np.max(RRI_time)-RRI_time[0])

        resamp_series=np.linspace(RRI_time[0],#開始時刻
                                  RRI_time[RRI_time.size-1],N
                                  )

        # 時間，RRI，fs(サンプリング周波数)
        return resamp_series,ff(resamp_series)

#LF :  0.04 ~ 0.15 Hz
#HF :  0.15 ~ 0.40 Hz
#central_period : 周期T
R_x = np.loadtxt("test1.csv",delimiter=",")
central_period = 1/(0.40+0.15) *2

resamp_series, signaldsgd = resamp_PPI(R_x,#心拍の山の時刻
               resamp_frequency=4.0,#リサンプリング周波数
               )
