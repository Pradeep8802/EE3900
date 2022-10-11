# # import numpy as np
# # np.fft.fft(np.exp(2j * np.pi * np.arange(8) / 8))


# # import matplotlib.pyplot as plt

# # t = np.arange(256)

# # sp = np.fft.fft(np.sin(t))

# # freq = np.fft.fftfreq(t.shape[-1])

# # plt.plot(freq, sp.real, freq, sp.imag)

# # plt.show()
# import numpy as np
# # np.fft.fft(np.exp(2j * np.pi * np.arange(8) / 8))
# import matplotlib.pyplot as plt

# def f(x):
#     # ytemp=np.array([1.0,2.0,3.0,4.0,2.0,1.0])
#     # y=np.pad(ytemp, (0,8), 'constant', constant_values=(0))
#     y=[1.0,2.0,3.0,4.0,2.0,1.0,0,0,0,0,0,0,0,0]
#     return y[x-1]

# x=np.array(14)
# z=[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
# y=[1.0,2.0,3.0,4.0,2.0,1.0,0,0,0,0,0,0,0,0]
# sp = np.fft.fft((y))

# freq = np.fft.fftfreq(14)
# # a=((sp.real**2 +sp.imag**2)**0.5)

# plt.plot(freq, sp.real,freq,sp.imag)

# plt.show()

# N = 14
# n = np.arange(N)
# fn=(-1/2)**n
# hn1=np.pad(fn, (0,2), 'constant', constant_values=(0))
# hn2=np.pad(fn, (2,0), 'constant', constant_values=(0))
# h = hn1+hn2
# X = np.zeros(N) + 1j*np.zeros(N)
# for k in range(0,N):
# 	for n in range(0,N):
# 		X[k]+=x[n]*np.exp(-1j*2*np.pi*n*k/N)

# t = np.arange(x)