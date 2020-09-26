# ###############################################################
# #
# Kevin Russell #
# ECE 351-51 #
# Lab 3 #
# September 15, 2020 #
# #
# #
# ###############################################################

#%% Part 1

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

def step(t): #defining step function using mathematical definition
    y=np.zeros(t.shape)
    
    for i in range (len(t)):
        if t[i]<0:
            y[i]=0
        else:
            y[i]=1
            
    return y    

def ramp(t): #defining ramp function using mathematical definition
    y=np.zeros(t.shape)
    
    for i in range (len(t)):
        if t[i]<0:
            y[i]=0
        else:
            y[i]=t[i]
    return y  

def f1(t):
    
    return step(t-2)-step(t-9)

def f2(t):
    
    return np.exp(t)

def f3(t):
    
    return ramp(t-2)*(step(t-2)-step(t-3))+ramp(4-t)*(step(t-3)-step(t-4))


plt.rcParams.update({'font.size':14})


steps =1e-2
t=np.arange(0,20+steps,steps)

f1=f1(t)
f2=f2(-t)
f3=f3(t)

plt.figure(figsize = (25,15))
plt.subplot(3,1,1)
plt.plot(t,f1)
plt.grid()
plt.ylabel('f1(t)')
plt.title('Part 1 User Defined Functions')

plt.subplot(3,1,2)
plt.plot(t,f2)
plt.grid()
plt.ylabel('f2(t)')

plt.subplot(3,1,3)
plt.plot(t,f3)
plt.grid()
plt.ylabel('f3(t)')


plt.xlabel('Time')
plt.show()


#%%Part 2

def conv(f_1,f_2): #Defining convolution
    Nf1=len(f_1) #defining length of the first function
    Nf2=len(f_2) #defining length of the second function
    
    f1new = np.append(f_1,np.zeros((1, (Nf2-1)))) #making both functions equal in length
    f2new = np.append(f_2,np.zeros((1, (Nf1-1))))
    result=np.zeros(f1new.shape) #creating array for output
    
    for i in range(Nf2+Nf1 -2): #for loop to go through all values of t (length of functions added together)
        result[i] =0
        for j in range (Nf1): #for loop to go through all values of the first function
            if(i - j + 1 > 0): #this multiplies the two functions together as the first passes by the second in a graphical sense
                result[i] += f1new[j]*f2new[i - j + 1]
            
    return result

steps = 1e-2
t=np.arange(0,20+steps, steps)
NN=len(t)
tnew = np.arange(0,2*t[NN-1],steps) #makes the length match the newly convolved functions


#%%%Task 2

conv12=conv(f1,f2)*steps
conv12check= sig.convolve(f1,f2)*steps

plt.figure(figsize=(10,7))
plt.plot(tnew, conv12, label ='User-Defined Convolution')
plt.plot(tnew, conv12check, '--', label ='Built-In Convolution')
plt.ylim([0,1.2])
plt.grid()
plt.xlabel('t[s]')
plt.ylabel('f1(t)*f2(t)')
plt.title('Convolution of f1 and f2')
plt.show()

#%%%Task 3
conv23 = conv(f2,f3)*steps
conv23check = sig.convolve(f2,f3)*steps

plt.figure(figsize=(10,7))
plt.plot(tnew, conv23, label ='User-Defined Convolution')
plt.plot(tnew, conv23check, '--', label ='Built-In Convolution')
plt.ylim([0,1.2])
plt.grid()
plt.xlabel('t[s]')
plt.ylabel('f2(t)*f3(t)')
plt.title('Convolution of f2 and f3')
plt.show()

#%%%Task 4
conv13 =conv(f1,f3)*steps
conv13check=sig.convolve(f1,f3)*steps

plt.figure(figsize=(10,7))
plt.plot(tnew, conv13, label ='User-Defined Convolution')
plt.plot(tnew, conv13check, '--', label ='Built-In Convolution')
plt.ylim([0,1.2])
plt.grid()
plt.xlabel('t[s]')
plt.ylabel('f1(t)*f3(t)')
plt.title('Convolution of f1 and f3')
plt.show()

