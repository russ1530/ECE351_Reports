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

def exp(t):
        
    return np.exp(t)

def cos(t, f):
    
    w = 2* np.pi * f
    
    return np.cos(w*t)

    

def h1(t):
    
    return (exp(2*t))*(step(1-t))

def h2(t):
    
    return step(t-2)-step(t-6)

def h3(t):
    
    return cos(t, 0.25)*step(t)


plt.rcParams.update({'font.size':14})


steps =1e-2
t=np.arange(-10,10+steps,steps)

f1=h1(t)
f2=h2(t)
f3=h3(t)

plt.figure(figsize = (25,15))
plt.subplot(3,1,1)
plt.plot(t,f1)
plt.grid()
plt.ylabel('h1(t)')
plt.title('Part 1 Signals')

plt.subplot(3,1,2)
plt.plot(t,f2)
plt.grid()
plt.ylabel('h2(t)')

plt.subplot(3,1,3)
plt.plot(t,f3)
plt.grid()
plt.ylabel('h3(t)')


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
t=np.arange(-10,10+steps, steps)
NN=len(t)
tnew = np.arange(2*t[0],2*t[NN-1]+steps,steps) #makes the length match the newly convolved functions

yy1 = step(t)
#%%%Python Calculated

conv1=conv(f1,yy1)*steps
conv2=conv(f2,yy1)*steps
conv3=conv(f3,yy1)*steps

plt.figure(figsize=(10,7))
plt.subplot(3,1,1)
plt.plot(tnew, conv1)
plt.grid()
plt.ylabel('h1(t)*u(t)')
plt.title('Python Impulse Response Calculation')

plt.subplot(3,1,2)
plt.plot(tnew, conv2)
plt.grid()
plt.ylabel('h2(t)*u(t)')

plt.subplot(3,1,3)
plt.plot(tnew, conv3)
plt.grid()
plt.xlabel('t[s]')
plt.ylabel('h3(t)*u(t)')

plt.show()

#%%%Hand Calculated



h1=(1/2*np.exp(2*tnew)*step(1-tnew))+(np.exp(2)*step(tnew-1))
h2=((tnew-2) * step (tnew-2)) - ((tnew-6)*step(tnew-6))
h3=0.6366 * np.sin(1.5708*tnew)*step(tnew)




plt.figure(figsize=(10,7))
plt.subplot(3,1,1)
plt.plot(tnew, h1)
plt.grid()
plt.ylabel('h1(t)*u(t)')
plt.title('Hand Impulse Response Calculation')

plt.subplot(3,1,2)
plt.plot(tnew, h2)
plt.grid()
plt.ylabel('h2(t)*u(t)')

plt.subplot(3,1,3)
plt.plot(tnew, h3)
plt.grid()
plt.xlabel('t[s]')
plt.ylabel('h3(t)*u(t)')

plt.show()