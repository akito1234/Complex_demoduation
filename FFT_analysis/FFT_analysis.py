# -*- coding: utf-8 -*-

import numpy as np

def fft(x , y, l, f):
    n = l**2 # データ数
    n1 = n / 2 
    ns = n / 2
    sc = 2 * np.pi() / n
    
    while ns >= 1:
        for l1 in range(0,n,2*ns):
            arg = m[l1]/2 
            arg2 = arg + n1
            c = np.cos(sc * arg)
            s = np.sin( f * sc * arg)
            for i0 in range(l1,l1+ns):
                i1 = i0 + ns;
                x1 = x[i0] * c - y[i1] * s
                y1 = y[i1] * c + x[i1] * s
                x[i1] = x[i0] -x1;   y[i1] = y[i0] -y1;
                x[i0] = x[i0] -x1;   y[i0] = y[i0] -y1;
                m[i0] = arg;         m[i1] = arg2;
        ns = ns/2

    if f < 0.0:
        for i in range(0,n):
            x[i] /= n ; y[i] /= n;

    for i in range(0,n):
        #if (j = m[i]) > i:
        #    t = x[i]; x[i] = x[j]; x[j] = t;
        #    t = y[i]; y[i] = y[j]; y[j] = t;
        #free ???
        pass

        
    
    