import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
def an(a,b,n):
    if n<0:
        return 0.0
    else:
        return (a**n-b**n)/(a-b)
def bn(a,b,n):
    if n>=1:
        return an(a,b,n-1)+an(a,b,n+1)
    else:
        return 0.0
def rhs(a,b,n):
    return an(a,b,n+2)-1
a=(1+math.sqrt(5))/2
b=(1-math.sqrt(5))/2


#1.1
n=np.arange(1,12)
vec_an=scipy.vectorize(an)

def f2(a,b,n):
    return np.sum(vec_an(a,b,np.arange(n)))
vec_rhs=scipy.vectorize(rhs)
vec_f2=scipy.vectorize(f2)
l1=vec_rhs(a,b,n)
l2=vec_f2(a,b,n)
#plt.subplot(211)
#plt.stem(n,l1,label=r'$a_{n+2}-1$')
#plt.grid()
#plt.legend()
#plt.subplot(212)
#plt.stem(n,l2,label=r'$\sum_{k=1}^{n}a_{k}$')
#pkadsf;kljasdfassssasdf;;l;lkjasdf;lkjadsfl;kj  adf;lk ;lkj qqwaerpawawqawerpoiuhgnghvnktyfm;thytlyhdropyhkuuuuugrgylt.grid()
#plt.legend()
#plt.savefig('../figs/1.1.png')
#plt.show()
# import numpy as np
# import matplotlib.pyplot as plt
# import math
# import scipy
# def an(a,b,n):
#     if n<0:
#         return 0.0
#     else:
#         return (a**n-b**n)/(a-b)
# def bn(a,b,n):
#     if n>=1:
#         return an(a,b,n-1)+an(a,b,n+1)
#     else:
#         return 0.0
# def rhs(a,b,n):
#     return an(a,b,n+2)-1
# a=(1+math.sqrt(5))/2
# b=(1-math.sqrt(5))/2


# #1.1
# n=np.arange(1,12)
# vec_an=scipy.vectorize(an)

def f2(a,b,n):
    return np.sum(vec_an(a,b,np.arange(n)))
vec_rhs=scipy.vectorize(rhs)
vec_f2=scipy.vectorize(f2)
l1=vec_rhs(a,b,n)
l2=vec_f2(a,b,n)
# 1.2
def f3(a,b,n):
   return np.dot(vec_an(a,b,np.arange(n)),np.array([1/10**i for i in range(n)]))
vec_f3=scipy.vectorize(f3)
x=np.linspace(0,12,12)
y=np.ones(12)*10/89
l3=vec_f3(a,b,n)
#plt.stem(n,l3,label=r'$\sum_{k=1}^{n}\frac{a_{k}}{10^k}$')
#plt.plot(x,y,label=r'10/89',color='orange')
#plt.legend()
#plt.grid()
#plt.savefig('../figs/1.2.png')
#plt.show()

#1.3
def f4(a,b,n):
    return a**n+b**n
vec_bn=scipy.vectorize(bn)
vec_f4=scipy.vectorize(f4)
l4=vec_bn(a,b,n)
l5=vec_f4(a,b,n)
# plt.subplot(211)
# plt.stem(n,l4,label=r'$b_{n}$')
# plt.grid()
# plt.legend()
# plt.subplot(212)
# plt.stem(n,l5,label=r'$\alpha^n+\beta^n$')
# plt.grid()
# plt.legend()
# #plt.savefig('../figs/1.3.png')
# plt.show()

#1.4
def f5(a,b,n):
   return np.dot(vec_bn(a,b,np.arange(n)),np.array([1/10**i for i in range(0,n)]))
vec_f5=scipy.vectorize(f5)
s=12
x=np.linspace(0,12,12)
y=np.ones(12)*8/89
l6=vec_f5(a,b,n)
plt.stem(n,l6,label=r'$\sum_{k=1}^{n}\frac{b_{k}}{10^k}$')
plt.plot(x,y,label=r'8/89',color='orange')
plt.grid()
plt.legend()
plt.savefig('../figs/1.4.png')
plt.show()


# import math as m
# import scipy as sc
# import numpy as np
# import matplotlib.pyplot as plt

# def an(a,b,n):
#     if n<=0:
#         return 0.0
#     if(n>=1):
#         return (a**n-b**n)/(a-b)

# def bn(a,b,n):
#     if n==1:
#         return 1.0
#     if(n>=1):
#         return an(a,b,n-1)+an(a,b,n+1)  

# a=(1+m.sqrt(5))/2
# b=(1-m.sqrt(5))/2        
# n=np.arange(1,100)
# vec_an=sc.vectorize(an)

# # 1.2
# def f(a,b,n):
#    return np.dot(vec_an(a,b,np.arange(n)),np.array([1/10**i for i in range(n)]))

# #1.1
# def lhs(a,b,n):
#     return an(a,b,n+2)-1

# #1.1
# def rhs(a,b,n):
#     return np.sum(vec_an(a,b,np.arange(n)))

# vec_rhs=sc.vectorize(rhs)
# vec_lhs=sc.vectorize(lhs)
# l1=vec_rhs(a,b,n)
# l2=vec_lhs(a,b,n)

# # plt.subplot(211)
# # plt.plot(n,l1,label=r'$a_{n+2}-1$',color='r')
# # plt.grid()
# # plt.legend()
# # plt.subplot(212)
# # plt.plot(n,l2,label=r'$\sum_{k=1}^{n}a_{k}$')
# # plt.grid()
# # plt.legend()
# # plt.show()

# #1.2
# # x1, y1 = [0,100], [10/89,10/89]
# # vec_f=sc.vectorize(f)
# # l3=vec_f(a,b,n)
# # plt.plot(n,l3,label=r'$\sum_{k=1}^{n}\frac{a_{k}}{10^k}$',color='orange')
# # plt.plot(x1, y1,color='b')
# # plt.legend()
# # plt.grid()
# # plt.show()

# #1.3
# def rhs3(a,b,n):
#     return a**n+b**n


# # vec_f4=sc.vectorize(rhs3)
# # l4=vec_bn(a,b,n)
# # l5=vec_f4(a,b,n)
# # plt.subplot(211)
# # plt.plot(n,l4,label=r'$b_{n}$',color='r')
# # plt.grid()
# # plt.legend()
# # plt.subplot(212)
# # plt.plot(n,l5,label=r'$\alpha^n+\beta^n$')
# # plt.grid()
# # plt.legend()
# # plt.show()

# vec_bn=sc.vectorize(bn)
# def f5(a,b,n):
#    return np.dot(vec_bn(a,b,np.arange(n)),np.array([1/10**i for i in range(n)]))-8/89
# vec_f5=sc.vectorize(f5)
# l6=vec_f5(a,b,n)
# plt.plot(n,l6,label=r'$\sum_{k=1}^{n}\frac{b_{k}}{10^k}-(\frac{8}{89})$')
# plt.grid()
# plt.legend()
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
def an(a,b,n):
    if n<0:
        return 0.0
    else:
        return (a**n-b**n)/(a-b)
def bn(a,b,n):
    if n>=1:
        return an(a,b,n-1)+an(a,b,n+1)
    else:
        return 0.0
def rhs(a,b,n):
    return an(a,b,n+2)-1
a=(1+math.sqrt(5))/2
b=(1-math.sqrt(5))/2


#1.1
n=np.arange(1,100)
vec_an=scipy.vectorize(an)

def f2(a,b,n):
    return np.sum(vec_an(a,b,np.arange(n)))
vec_rhs=scipy.vectorize(rhs)
vec_f2=scipy.vectorize(f2)
l1=vec_rhs(a,b,n)
l2=vec_f2(a,b,n)
plt.subplot(211)
plt.plot(n,l1,label=r'$a_{n+2}-1$',color='r')
plt.grid()
plt.legend()
plt.subplot(212)
plt.plot(n,l2,label=r'$\sum_{k=1}^{n}a_{k}$')
plt.grid()
plt.legend()
plt.show()
#1.2
def f3(a,b,n):
   return np.dot(vec_an(a,b,np.arange(n)),np.array([1/10**i for i in range(n)]))
vec_f3=scipy.vectorize(f3)
x=np.linspace(0,100,1000)
y=np.ones(1000)*10/89
l3=vec_f3(a,b,n)
plt.plot(n,l3,label=r'$\sum_{k=1}^{n}\frac{a_{k}}{10^k}$',color='r')
plt.plot(x,y,label=r'10/89',color='b')
plt.legend()
plt.grid()
plt.show()
#1.3
def f4(a,b,n):
    return a**n+b**n
vec_bn=scipy.vectorize(bn)
vec_f4=scipy.vectorize(f4)
l4=vec_bn(a,b,n)
l5=vec_f4(a,b,n)
plt.subplot(211)
plt.plot(n,l4,label=r'$b_{n}$',color='r')
plt.grid()
plt.legend()
plt.subplot(212)
plt.plot(n,l5,label=r'$\alpha^n+\beta^n$')
plt.grid()
plt.legend()
plt.show()

#1.4
def f5(a,b,n):
   return np.dot(vec_bn(a,b,np.arange(n)),np.array([1/10**i for i in range(0,n)]))
vec_f5=scipy.vectorize(f5)
x=np.linspace(0,100,1000)
y=np.ones(1000)*8/89
l6=vec_f5(a,b,n)
plt.plot(n,l6,label=r'$\sum_{k=1}^{n}\frac{b_{k}}{10^k}$',color='r')
plt.plot(x,y,label=r'8/89')
plt.grid()
plt.legend()
plt.show()