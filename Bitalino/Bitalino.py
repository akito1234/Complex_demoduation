
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:58:24 2019

@author: akito
"""

# Import packages
from pyhrv.hrv import hrv
from opensignalsreader import OpenSignalsReader
from biosppy.signals.ecg import ecg
from biosppy.signals.eda import eda
from biosppy.signals.resp import resp
from  Sensor_Scale import EDA,PZT,ECG

import pyhrv.tools as tools
from pyhrv.frequency_domain import welch_psd

import matplotlib.pyplot as plt
import numpy as np
# Specify the file path of your OpenSignals file (absolute file path is recommended)
fpath = r"Z:\00_個人用\東間\theme\柴田_実験\opensignals_201806130003_2019-06-10_15-54-27.txt"

log_data = np.loadtxt(fpath)
# Acquire EDA PZT
# EDA : Analog1 10bit
# PZT : Analog2 10bit
signal = log_data[:,5]
#スケーリング
#signal_EDA = EDA(signal[:,0],10)
#signal_PZT = PZT(signal[:,1],10)
signal_ECG = ECG(signal,10)
#特徴量算出
#EDA_features = eda(signal_EDA,1000,show=False)
#RESP_features = resp(signal_PZT,1000,show=False)
#print(RESP_features[3] )
#np.savetxt(fname = r"Z:\00_個人用\東間\theme\Bitalino\センサ\PIZ_EDA\feature.csv"
#           ,X = RESP_features[4]
#           ,delimiter=",")
signal, rpeaks = ecg(signal_ECG, show=False)[1:3]
nni = tools.nn_intervals(rpeaks.tolist())
np.savetxt(fname = r"Z:\00_個人用\東間\theme\柴田_実験\RRI_list.csv"
           ,X = nni
           ,delimiter=",")
#output = welch_psd(nni,nfft=2**10,show=True)
# Compute NNI

#fig,axes = plt.subplots(2,1,sharex=True,figsize=(10,8))
#axes[0].plot(signal_EDA)
#axes[0].set_title("EDA")
#axes[0].set_ylabel("[us]")
#axes[1].plot(signal_PZT)
#axes[1].set_title("RESP")
#axes[1].set_ylabel("[%]")
