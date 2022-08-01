# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 16:00:13 2021

@author: Talebf
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import norm 

#######################################
def thz_pulse(t, tau=0.15, loc=0):
    return t * norm.pdf(t, scale=tau)

def thz_pulse2(t, tau=0.15):
    return t/tau/tau*(np.exp(-t**2/tau**2))

tau = 0.3 ##mm/ps
dt = 0.05 ##Ps
t = np.arange(0, 200, dt)  # Zeitachse
t_r = 46.45
t_0 = t_r + 0
n0 = 1
d1 = 3
n1 = 1.57
d2 = 0.25
n2 = 1.78
d3 = 0.01
n3 = 2.97
d4 = 3.292
n4 = 2.5
t_1 = 2*(d1*n1)/0.3 + t_0
ab1 = np.exp(-0.5)
t_2 = 2*(d2*n2)/0.3 + t_1
ab2 = np.exp(-0.5)
t_3 = 2*(d3*n3)/0.3 + t_2
ab3 = np.exp(-0.5)
t_4 = 2*(d4*n4)/0.3 + t_3
ab4 = 0.5*np.exp(-0.5)
print("1-layer:",t_1-t_0,"2-layer:",t_2-t_1,"3-layer:",t_3-t_2,"4-layer:",t_4-t_3)
r0 = (n0-n1)/(n0+n1)
t01 = 2*n0/(n0+n1)
t10 = 2*n1/(n0+n1)
r1 = (n1-n2)/(n1+n2)
t12 = 2*n1/(n1+n2)
t21 = 2*n2/(n1+n2)
r2 = (n2-n3)/(n2+n3)
t23 = 2*n2/(n2+n3)
t32 = 2*n3/(n2+n3)

r3 = (n3-n4)/(n3+n4)
t34 = 2*n3/(n3+n4)
t43 = 2*n4/(n4+n3)
r4 = -1

y = r4*thz_pulse2(t - t_r, tau)  # Referenz
y2 = r0*thz_pulse2(t - t_0 , tau) + ab1*t01*r1*t10*thz_pulse2(t - t_1 , tau) + ab1*ab2*t01*t12*r2*t10*t21*thz_pulse2(t - t_2, tau) + ab1*ab2*ab3*t01*t12*t23*r3*t32*t21*t10*thz_pulse2(t - t_3, tau)+ab1*ab2*ab3*ab4*t01*t12*t23*t34*r4*t43*t32*t21*t10*thz_pulse2(t - t_4, tau)
 
# Noise
# y  += np.random.random(len(t)) * np.max(np.abs(y))
y2 += np.random.random(len(t)) * np.max(np.abs(y2)) * 0.02
y += np.random.random(len(t)) * np.max(np.abs(y)) * 0.02

Y = np.abs(np.fft.fft(y))
freq = np.fft.fftfreq(len(t),dt)
idx = freq>0
plt.figure()
plt.plot(freq[idx],20*np.log10(Y[idx]))
plt.show()

plt.figure()
plt.plot(t,y,t,y2)
plt.show()


dx = 1
dy = 1
tilt = 15/180*np.pi
k = 0

name="Reference1.txt"
temp = np.vstack((t,y)).T
np.savetxt(name, temp)

for i in range(1):
    for j in range(1):
        k += 1
        t_2 = 2*(d2*n2)/0.3 + t_1 + np.tan(tilt)*i*dx/0.3 
        y2 = r0*thz_pulse2(t - t_0 , tau) + ab1*t01*r1*t10*thz_pulse2(t - t_1 , tau) + ab1*ab2*t01*t12*r2*t10*t21*thz_pulse2(t - t_2, tau) + ab1*ab2*ab3*t01*t12*t23*r3*t32*t21*t10*thz_pulse2(t - t_3, tau)+ab1*ab2*ab3*ab4*t01*t12*t23*t34*r4*t43*t32*t21*t10*thz_pulse2(t - t_4, tau)
        y2 += np.random.random(len(t)) * np.max(np.abs(y2)) * 0.02
        temp = np.vstack((t,y2)).T
        # lt = time.localtime()
        # tmstemp = str(lt[0]) + '-' + str(lt[1])+ '-' + str(lt[2])+ '-' + str(lt[3])+ 'T' + str(lt[4])+ '-' + str(lt[5])
        # name = tmstemp + '-NAME-[' + str(k) + ']-[' + str(i*dx) + ',' + str(j*dy) + ',' + '0]-[1.0,0.0,0.0,0.0]-delta[0.014mm-0.0deg]-avg20.txt'
        name ="sample1.txt"
        np.savetxt(name, temp)

#lt = time.localtime()
# tmstemp = str(lt[0]) + '-' + str(lt[1])+ '-' + str(lt[2])+ '-' + str(lt[3])+ 'T' + str(lt[4])+ '-' + str(lt[5])
#name = tmstemp + '-Reference-[' + str(0) + ']-[' + str(0) + ',' + str(0) + ',' + '0]-[1.0,0.0,0.0,0.0]-delta[0.014mm-0.0deg]-avg20.txt'
