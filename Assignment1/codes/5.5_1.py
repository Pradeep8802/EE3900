import numpy as np
import matplotlib.pyplot as plt



def delta(n):
    if n==0:
        return 1
    else:
        return 0
        
length=10
h=np.zeros(length)
h[0]=1

for i in range(1,length):
    h[i]=-(h[i-1]/2)+delta(i)+delta(i-2)

x=[1,2,3,4,2,1]
topizh=np.zeros([len(h)+len(x)-1,len(x)])

n=np.linspace(0,len(h)+len(x)-2,len(h)+len(x)-1)
print(n)
for i in range(len(x)):
    for j in range(len(h)):
        topizh[j+i][i]=h[j]

# y=np.convolve(h[0:10],x)

y=np.dot(topizh,x)

plt.stem(n,y)
plt.ylabel("$y(n)$")
plt.xlabel("$n$")
plt.grid()
plt.show()