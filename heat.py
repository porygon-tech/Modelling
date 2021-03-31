import sys
import numpy as np 
import math
import matplotlib.pyplot as plt  
#import matplotlib.animation as animation 

#python3 heat.py

def heatpropagate(U, delta_x, delta_t):
	height = U.shape[0] - 1
	width  = U.shape[1] - 1
	for i in range(0,height):    #time
		for j in range(1,width): #space
			UxL = (U[i,   j] - U[i, j-1]) / delta_x
			UxR = (U[i, j+1] - U[i,   j]) / delta_x
			Uxx = (UxR - UxL) #second derivative in x
		#	print(i,j,Uxx)
			U[i+1,j] = U[i,j] + k(x0+j*delta_x, t0+i*delta_t) * Uxx * delta_t
		U[i+1,0] = U[i+1,1]
		U[i+1,-2] = U[i+1,-1]
	return U

def heatframe(mat, t):
	a=mat[t,:]
	x = np.arange(mat.shape[1])
	plt.plot(x, a, color = 'red')
	plt.show()

'''
import pandas as pd
def heatframe(mat, t):
	a = pd.Series(mat[t,:])
	a.plot()
	plt.show()
'''

def showdata(mat, color=plt.cm.hot):
	mat = np.copy(mat)
	mat[0,0] = 0
	mat[0,1] = 1
	plt.imshow(mat.astype('float32'), interpolation='none', cmap=color)
	plt.show()



#----------PARAMETERS-----------

x0 = -10
x1 = 10

t0 = 0
t1 = 8

delta_x = 1/10
delta_t = 1/30

def f(x): #initial heat distribution
	if abs(x) < math.pi/2:
		return 1
 		#return math.cos(x)**2
	else:
		return 0

def k(x,t): #heat conductivity
	if x>0:
		return 1
	else:
		return 0.1

#----------MAIN-----------------

height = int((t1-t0)/delta_t)
width  = int((x1-x0)/delta_x)

U = np.zeros((height+1,width+1)) 
for i in range(width+1):
	U[0,i] = f(x0+(i-1)*delta_x)

U[0,:] = np.random.rand(width+1)



heatpropagate(U,delta_x, delta_t)
showdata(U)
heatframe(U,0)
heatframe(U,-1)




