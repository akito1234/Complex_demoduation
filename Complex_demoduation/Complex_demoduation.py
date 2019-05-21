#*---config:utf-8---------
#
#refernce :https://currents.soest.hawaii.edu/ocn_data_analysis/_static/complex_demod.html
#
#

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.base import Bunch
from scipy import signal,interpolate,integrate
import math

def bl_filt(y, half_width):
    """
    Simple Blackman filter.
    
    The end effects are handled by calculating the weighted
    average of however many points are available, rather than
    by zero-padding.
    
    最終効果は、ゼロパディングではなく、多くのポイントの加重平均を
    計算することによって処理されます。
    """

    # 段階数
    nf = half_width * 2 + 1
    x = np.linspace(-1, 1, nf, endpoint=True)
    x = x[1:-1]   # chop off the useless endpoints with zero weight
    #-1と1を取り除く
    w = 0.42 + 0.5 * np.cos(x * np.pi) + 0.08 * np.cos(x * 2 * np.pi)
    ytop = np.convolve(y, w, mode='same')
    
    ybot = np.convolve(np.ones_like(y), w, mode='same')
    #np.savetxt(r"C:\Users\akito\Desktop\result2.csv",            # ファイル名
    #           X=ybot.real,                  # 保存したい配列
    #           delimiter=","            # 区切り文字
    #)
    #np.savetxt(r"C:\Users\akito\Desktop\Hashimoto\output.csv",            # ファイル名
    #       X=(ytop / ybot).real,                  # 保存したい配列
    #       delimiter=","            # 区切り文字
    #       )
 
    return ytop / ybot

def test_data(periods, 
              noise=0,
              rotary=False, 
              npts=1000,
              dt=1.0/24):
    """
    Generate a simple time series for testing complex demodulation.
    
    *periods* is a sequence with the periods of one or more
        harmonics that will be added to make the test signal.
        They can be positive or negative.
        高周波振動数
    *noise* is the amplitude of independent Gaussian noise.    
    *rotary* is Boolean; if True, the test signal is complex.
    *npts* is the length of the series.
    *dt* is the time interval (default is 1.0/24)
    
    Returns t, x: ndarrays with the test times and test data values.
    
    Note: the default of dt = 1/24 corresponds to hourly values in
    units of days, so a period of 12.42/24 would give the M2 frequency.
    """     
        
    t = np.arange(npts, dtype=float) * dt
    
    if rotary:
        x = noise * (np.random.randn(npts) + 1j * np.random.randn(npts))
    else:
        x = noise * np.random.randn(npts)
    #角周波数での波長の和を出す
    for p in periods:
        if rotary:
            x += np.exp(2j * np.pi * t / p)
        else:
            x += np.cos(2 * np.pi * t / p)
    
    return t, x


def complex_demod(t, x, central_period, hwidth = 2):
    """
    それでは、1-D入力に対する基本的な複素復調の実装を提供します。
    一般に、一度に1つの回転成分しか得られないため、
    複雑な入力に対しては2回呼び出す必要があります。
    さらに数行のコードで、1回の呼び出しで両方のコンポーネントを生成するように
    変更できます。それでは、1-D入力に対する基本的な複素復調の実装を提供します。
    一般に、一度に1つの回転成分しか得られないため、複雑な入力に対しては2回
    呼び出す必要があります。さらに数行のコードで、
    1回の呼び出しで両方のコンポーネントを生成するように変更できます。
    Complex demodulation of a real or complex series, *x*
    of samples at times *t*, assumed to be uniformly spaced.

    等間隔であると仮定された、時間* t *における
    実数または複素数* x *のサンプルの複素復調。

    *central_period* is the period of the central frequency
        for the demodulation.  It should be positive for real
        signals. For complex signals, a positive value will
        return the CCW rotary component, and a negative value
        will return the CW component (negative frequency).
        Period is in the same time units as are used for *t*.

    * central_period *は復調の中心周波数の周期です。実際の信号に対してはプラスになります。
        複素数信号の場合、正の値はCCW回転成分を返し、負の値はCW成分（負の周波数）を返します。
        Periodは* t *に使用されるのと同じ時間単位です
        反時計回りをCCW
        時計回りはCW

    *hwidth* is the Blackman filter half-width in units of the 
        *central_period*.  For example, the default value of 2
        makes the Blackman half-width equal to twice the 
        central period.
    
    * hwidth *はBlackmanフィルタの半値幅で、* central_period *の単位です。
        たとえば、デフォルト値の2は、Blackmanの半値幅を中央期間の2倍にします。
    
    Returns a Bunch; look at the code to see what it contains.
    """     
    #xが複素浮動小数の場合rotary=True
    rotary = x.dtype.kind == 'c'  # complex input
    
    # Make the complex exponential for demodulation:
    #復調の複素指数関数を作る：
    c = np.exp(-1j * 2 * np.pi * t / central_period)
    
    product = x * c
    
    # filter half-width number of points
    dt = t[1] - t[0]
    # 1周期に取れるデータ数
    hwpts = int(round(hwidth * abs(central_period) / dt))
    
    # apply filter
    demod = bl_filt(product, hwpts)

    if not rotary:    
        # The factor of 2 below comes from fact that the
        # mean value of a squared unit sinusoid is 0.5.
        # 以下の2の要素は、二乗された単位正弦波の平均値は0.5です。
        demod *= 2
    # np.conjで共役複素数 (複素共役, 虚数部の符号を逆にした複素数) を返す

    reconstructed = (demod * np.conj(c))
    
    if not rotary:
        reconstructed = reconstructed.real
        
    if np.sign(central_period) < 0:
        demod = np.conj(demod)
        # This is to make the phase increase in time
        # for both positive and negative demod frequency
        # when the frequency of the signal exceeds the
        # frequency of the demodulation.
    
    return Bunch(t=t,
                 signal=x,  
                 hwpts=hwpts,
                 demod=demod,
                 reconstructed=reconstructed)


def plot_demod(dm):
    freq = np.linspace(0, 4.0, 1194)

    fig, axs = plt.subplots(3, sharex=False)
    resid = dm.signal - dm.reconstructed
    if dm.signal.dtype.kind == 'c':
        axs[0].plot(dm.t, dm.signal.real, label='signal.real')
        axs[0].plot(dm.t, dm.signal.imag, label='signal.imag')
        axs[0].plot(dm.t, resid.real, label='difference real')
        axs[0].plot(dm.t, resid.imag, label='difference imag')
    else:    
        #axs[0].plot(freq, function(dm.signal), label='signal')
        #axs[0].plot(freq, function(dm.reconstructed), label='reconstructed')
        #axs[0].set_xlim(0,1)
        #axs[0].set_ylim(0,200)
        axs[0].plot(dm.t, dm.signal, label='signal')
        #axs[0].plot(dm.t, dm.reconstructed+np.average(dm.signal), label='reconstructed')
        axs[0].plot(dm.t, dm.reconstructed, label='reconstructed')
        #axs[0].plot(dm.t, dm.signal - dm.reconstructed, label='difference')
    
    axs[0].legend(loc='upper right', fontsize='small')
    
    axs[1].plot(dm.t, np.abs(dm.demod), label='amplitude', color='C3')
    axs[1].legend(loc='upper right', fontsize='small') 
    
    axs[2].plot(dm.t, np.angle(dm.demod, deg=True), '.', label='phase',
                color='C4')
    axs[2].set_ylim(-180, 180)
    axs[2].legend(loc='upper right', fontsize='small')
    
    for ax in axs:
        ax.locator_params(axis='y', nbins=5)
    return fig, axs    

def function(signal):
    F = np.fft.fft(signal)
    ## 正規化 + 交流成分2倍
    #F = F/(N/2)
    #F[0] = F[0]/2
    # 振幅スペクトルを計算
    Amp = np.abs(F)
    Power = Amp ** 2
    return Power


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

def main_dmod(R_x,
              central_period,
              resamp_frequency=2.0,
              hwidth = 2):

    t,x = resamp_PPI(R_x,resamp_frequency)
    dm = complex_demod(t, x, central_period, hwidth=hwidth)
    fig, axs = plot_demod(dm)
    return fig, axs,dm
 
def test_demod(periods, #周期(配列)
               central_period,
               noise=0,
               rotary=False, 
               hwidth = 1, 
               npts=1000,
               dt=1.0/24):
    # t が時間
    # x がテストデータ
    t, x = test_data(periods, noise=noise, rotary=rotary, 
                     npts=npts, dt=dt)
    dm = complex_demod(t, x, central_period, hwidth=hwidth)
    fig, axs = plot_demod(dm)
    return fig, axs, dm


    
#12.0/24, 13.0/24, 14.5/24の3つの周期を選択
#central_period(取り出す周期)は12.0/24.0
#test_demod([12.0/24, 20.0/24, 14.5/24], 12.0/24,noise=1);

#LF :  0.04 ~ 0.15 Hz
#HF :  0.15 ~ 0.40 Hz
#central_period : 周期T
R_x = np.loadtxt("test1.csv",delimiter=",")
central_period = 1/(0.40+0.15) *2
fig, axs,dm = main_dmod(R_x,central_period,resamp_frequency=4.0,hwidth=2.0)
resamp_series,signaldsgd = resamp_PPI(R_x,#心拍の山の時刻
               resamp_frequency=4.0,#リサンプリング周波数
               )
freq = np.linspace(0, 4.0, 1194)

#plt.figure()
#plt.plot(freq,function(dm.signal))
#plt.xlim(0,1)
##plt.plot(t,RRI)
plt.show()
