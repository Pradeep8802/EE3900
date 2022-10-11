
import matplotlib.pyplot as plt

def add(a,b):
    return (a+b)

def add3(a,b,c):
    return (add(add(a,b),c))

def show(a):
    print(a)

show(add3(1,2,3))





