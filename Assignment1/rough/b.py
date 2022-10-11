import numpy as np
import matplotlib.pyplot as plt
#signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)
signal = np.array([1.0,2.0,3.0,4.0,2.0,1.0,0,0,0,0,0,0,0,0])
fourier = np.fft.fft(signal)
n = signal.size
timestep = 0.1
freq = np.fft.fftfreq(n, d=timestep)
# plt.plot(freq, fourier.real,freq,fourier.imag)
s=(fourier.real**2+fourier.imag**2)**0.5
plt.plot(freq, s)
plt.show()

