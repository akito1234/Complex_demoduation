
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:58:24 2019

@author: akito
"""

import numpy as np
def EDA(raw,bit):
    Vcc = 3.3 #電源電圧3.3V
    # us [マイクロセカンド出力]
    return raw/(2**bit) * Vcc / 0.132
    pass
def PZT(raw,bit):
    # [%]
    return (raw/2**bit-0.5)*100
    pass
def ECG(raw,bit):
    Vcc = 3.3 #電源電圧3.3V
    Gain = 1100
    # [V]
    return (raw/2**bit-0.5)*Vcc/Gain
    pass