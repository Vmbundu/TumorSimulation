import math

#Functions used for mapping in Vasculature3D.py
def z1_fuc(x):
    return 200 * (math.e ** (-(x ** 2)/(2*(380**2))))
def z2_fuc(x,y):
    return 300 * math.e ** (-((x-450)**2+(y-480)**2)/(2*300**2))-60
def z3_fuc(x):
    return 200 * math.e ** (-(x-1050)**2/(2*100**2))
def z4_fuc(y):
    return 200 * math.e ** (-(y-1080)**2/(2*100**2))
def z5_fuc(x,y):
    return -150*math.e ** (-((x-1000)**2+(y-1000)**2)/(2*100**2))
