# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 13:59:21 2017

@author: keita
"""

#============function==============
import sys,os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal,interpolate,integrate
import math


class PSD:
    def resample(self,R_x,N):
        RRI=np.zeros(R_x.size-1)
        RRI_time=np.zeros(R_x.size-1)
        #山の間隔を秒に変換する
        for i in range(R_x.size-1):
            RRI[i]=(R_x[i+1]-R_x[i])*0.001
            RRI_time[i]=R_x[i+1]*0.001
    
        # 3 次スプライン補間
        # RRI:山の間隔
        # RRI_time : 時間軸
        # 120sを512で分割
        # 約4Hzでリサンプリングする
        ff = interpolate.interp1d(RRI_time,RRI,kind='cubic')

        resamp_series=np.linspace(RRI_time[0],#開始時刻
                                  RRI_time[RRI_time.size-1],#120秒のデータ
                                  N#分割数
                                  )
        #サンプリング周波数
        resamp_frequency=N/(np.max(RRI_time)-RRI_time[0])

        return ff(resamp_series),resamp_frequency


    def cal_PSD(self,RRI_time,RRI):
        ff=interpolate.interp1d(RRI_time,RRI,kind='cubic')
        # PSDを出力する
        # freq -> 周波数
        # P1 ->　パワースペクトル
        freq1,P1=signal.welch(ff(resamp_series),#リサンプリングされたRRI時系列
                              resamp_frequency,#サンプリング周波数
                              nperseg=N #セグメント長
                              )
        # 線形補間
        inter_PSD=interpolate.interp1d(freq1,P1)
        pass

    def freq_PSD(self,
                 R_x,#心拍の山の時刻
                 N,#データ数
                 LF_right,#LF帯の最大値:0.15
                 HF_left,#HF帯の最小値:0.15
                 HF_right#HF帯の最大値:0.40
                 ):
        #[1*120]
        RRI=np.zeros(R_x.size-1)
        RRI_time=np.zeros(R_x.size-1)
        #山の間隔を秒に変換する
        for i in range(R_x.size-1):
            RRI[i]=(R_x[i+1]-R_x[i])*0.001
            RRI_time[i]=R_x[i+1]*0.001
    
        # 3 次スプライン補間
        # RRI:山の間隔
        # RRI_time : 時間軸
        # 120個のデータを512個に増やす
        # 約4Hzでリサンプリングする
        ff=interpolate.interp1d(RRI_time,RRI,kind='cubic')

        resamp_series=np.linspace(RRI_time[0],#開始時刻
                                  RRI_time[RRI_time.size-1],N
                                  #120秒のデータ
                                  )

        #サンプリング周波数
        resamp_frequency=N/(np.max(RRI_time)-RRI_time[0])

        



        #---------------フーリエ変換-------------------#
        F = np.fft.fft(ff(resamp_series))
        ## 正規化 + 交流成分2倍
        #F = F/(N/2)
        #F[0] = F[0]/2
        # 振幅スペクトルを計算
        Amp = np.abs(F)
        Pow = Amp ** 2
        freq = np.linspace(0, resamp_frequency, N) # 周波数軸

        #-------------------LF-------------------------#
        # 低周波領域
        # 元波形をコピーする
        LF = F.copy()
        LF[((freq < 0.04)|(freq > 0.15))] = 0 + 0j
        Amp_LF = np.abs(LF)
        Pow_LF = Amp_LF ** 2
        # パワースペクトルの計算（振幅スペクトルの二乗）
        # 高速逆フーリエ変換
        lf = np.fft.ifft(LF)

        # 実部の値のみ取り出し
        lf = lf.real
        LFA =  np.abs(lf)

        #-------------------HF------------------------#
        # 高周波領域
        # 元波形をコピーする
        HF = F.copy()
        freq = np.linspace(0, resamp_frequency, N) # 周波数軸
        # ローパスフィル処理（カットオフ周波数を超える帯域の周波数信号を0にする）
        HF[((freq < 0.15)|(freq > 0.40))] = 0 + 0j

        Amp_HF = np.abs(HF)
        Pow_HF = Amp_HF ** 2
        # パワースペクトルの計算（振幅スペクトルの二乗）
        # 高速逆フーリエ変換
        hf = np.fft.ifft(HF)
        # 実部の値のみ取り出し
        hf = hf.real
        HFA =  np.abs(hf)

        #---------------パワースペクトルを計算-------------------#
        # PSDを出力する
        # freq -> 周波数
        # P1 ->　パワースペクトル

        freq1,P1=signal.welch(ff(resamp_series),#リサンプリングされたRRI時系列
                              resamp_frequency,#サンプリング周波数
                              nperseg=N #セグメント長
                              )
        # 線形補間
        inter_PSD=interpolate.interp1d(freq1,P1)
        #np.argmax : 指定された配列の中で最大値となっている要素のうち先頭のインデックスを返します。
        LF_i=np.argmax(freq1>=0.04)
        LF_iend=np.argmax(freq1>=LF_right)
        #freq  ???
        LF_x0=np.r_[0.04,freq1[LF_i:LF_iend]]
        LF_x=np.r_[LF_x0,LF_right]
        #psd   ???
        LF_y0=np.r_[inter_PSD(np.array([0.04])),P1[LF_i:LF_iend]]
        LF_y=np.r_[LF_y0,inter_PSD(np.array([LF_right]))]
        LF_psd=integrate.simps(LF_y,LF_x)*1.0e6


        
        output = ff(resamp_series)
        #np.savetxt('test2.csv',output,delimiter=',')
        #plt.figure()
        ## plt.plot(freq1, 10*np.log10(P1), "b")
        ##plt.plot(freq, Pow, "b")
        #plt.plot(resamp_series, HFA, "b")
        ## plt.plot(LF_x,LF_y, "b")
        ##plt.plot(resamp_series, hfa , "b")
        ##plt.xlim(0,1)
        #plt.ylim( 0,10)
        #plt.legend(loc="upper right")
        #plt.xlabel("Frequency[Hz]")
        #plt.ylabel("Power/frequency[dB/Hz]")


        #fig, axs = plt.subplots(3, 1,sharex=True ,figsize=(12, 8))
        ## 左上
        ## orginal wave
        #axs[0].plot(resamp_series, ff(resamp_series)*1000 , "b")
        #axs[0].set_ylabel("RRI[ms]")
        #axs[0].set_ylim(800,1400)
        ## LF wave
        #axs[1].plot(resamp_series,  lf*1000 , "b")
        #axs[1].set_ylabel("LF[ms]")
        #axs[1].set_ylim(-100,100)
        
        ## 左下
        #axs[2].plot(resamp_series,  hf*1000 , "b")
        #axs[2].set_ylabel("HF[ms]")
        #axs[2].set_ylim(-100,100)
        #plt.xlabel("Time[s]")
        ## 右下
        ## axs[1, 1].plot(LF_x,LF_y, "b")

        
        fig, axs = plt.subplots(2, 1,sharex=True ,figsize=(12, 8))
        # 左上
        # LF wave
        axs[0].plot(resamp_series,  LFA*1000 , "b")
        axs[0].set_ylabel("LFA[ms]")
        axs[0].set_ylim(0,100)
        
        # 左下
        axs[1].plot(resamp_series,  HFA*1000 , "b")
        axs[1].set_ylabel("HFA[ms]")
        axs[1].set_ylim(0,100)
        plt.xlabel("Time[s]")
        # 右下
        # axs[1, 1].plot(LF_x,LF_y, "b")
        

        #fig, axs = plt.subplots(3, 3, figsize=(12, 8))
        ## 左上
        ## orginal wave
        #axs[0,1].plot(resamp_series, ff(resamp_series) , "b")
        #axs[0,2].plot(freq, Pow, "b")
        #axs[0,2].set_ylim(0,50)
        ## LF wave
        #axs[1,0].plot(resamp_series,  LFA , "b")
        #axs[1,1].plot(resamp_series, lf, "b")
        ##axs[1,1].set_ylim(0,10)
        ##axs[1,1].set_xlim(0,1)
        #axs[1,2].plot(freq, Pow_LF, "b")
        ## 左下
        #axs[2,0].plot(resamp_series,  HFA , "b")
        #axs[2,1].plot(resamp_series, hf, "b")
        ##axs[2,1].set_ylim(0,10)
        ##axs[2,1].set_xlim(0,1)
        #axs[2,2].plot(freq, Pow_HF, "b")
        ## 右下
        ## axs[1, 1].plot(LF_x,LF_y, "b")


        # 左上
        #axs[0,0].plot(resamp_series, hf , "b")
        #axs[0,1].plot(freq, Pow, "b")
        #axs[0,1].set_ylim(0,50)
        # 右上
        #axs[1,0].plot(resamp_series,  lf , "b")
        
        #axs[1,1].plot(freq,LF, "b")
        #axs[1,1].set_ylim(0,10)
        #axs[1,1].set_xlim(0,1)
        # 左下
        #axs[2,0].plot(resamp_series, ff(resamp_series) , "b")
        #axs[2,1].plot(freq, HFA, "b")
        #axs[2,1].set_ylim(0,10)
        #axs[2,1].set_xlim(0,1)
        # 右下
        # axs[1, 1].plot(LF_x,LF_y, "b")


        plt.show()
        return 0 



R_x = np.loadtxt("test1.csv",delimiter=",")
freq = PSD()
#signal = freq.freq_PSD(R_x,R_x.size*512/120,0.15,0.15,0.4)
signal = freq.freq_PSD(R_x[0:300], 512*300/120 ,0.15,0.15,0.4)
#RRI算出

#Freq=freq_PSD(R[1][start:Index],512,0.15,0.15,0.4)
#features[i,np.arange(4)]=np.array([Freq[0],Freq[1],Freq[2],Freq[3]])
