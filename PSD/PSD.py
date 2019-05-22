import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

n = 1024
dt = 0.001
fs = 1/dt
f1 = 120
f2 = 150
t = np.linspace(1, n, n)*dt-dt
y = np.sin(2*np.pi*f1*t)+2*np.sin(2*np.pi*f2*t)+0.1*np.random.randn(t.size)

freq1, P1 = signal.periodogram(y, fs)
freq2, P2 = signal.welch(y, fs)
freq3, P3 = signal.welch(y, fs, nperseg=n/2)
freq4, P4 = signal.welch(y, fs, nperseg=n/8)

plt.figure()
plt.plot(freq1, 10*np.log10(P1), "b", label="periodogram")
plt.plot(freq2, 10*np.log10(P2), "r", linewidth=2, label="nseg=n/4")
plt.plot(freq3, 10*np.log10(P3), "c", linewidth=2, label="nseg=n/2")
plt.plot(freq4, 10*np.log10(P4), "y", linewidth=2, label="nseg=n/8")
plt.ylim(-60, 0)
plt.legend(loc="upper right")
plt.xlabel("Frequency[Hz]")
plt.ylabel("Power/frequency[dB/Hz]")
# plt.show()
