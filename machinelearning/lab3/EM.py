import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataSrc = ['data/data_1_large.txt', 'data/data_1_small.txt', 'data/data_2_large.txt', 
		'data/data_2_small.txt','data/data_3_large.txt', 'data/data_3_small.txt']
dataMystery = ['data/mystery_1.txt', 'data/mystery_2.txt', 'data/mystery_3.txt']
data = dataSrc + dataMystery

def main():
	softEM()
	hardEM()

def softEM( iterations = 10 ):
	X = np.array( readFile(data[8]) )	# Chose the data file
	MU = initMu( X )
	SIGMA = initSigma( X, MU )
	posterior = initPosterior()
	# Iterate
	for _ in range( iterations ):
		currentP = e_step( X, MU, SIGMA, posterior )
		MU, SIGMA, posterior = m_step( X, currentP )

	plot( X, MU, SIGMA, currentP )

def hardEM( iterations = 10 ):
	X = np.array( readFile(data[8]) )	# Chose the data file
	MU = initMu( X )
	SIGMA = initSigma( X, MU )
	posterior = initPosterior()
	# Iterate
	for _ in range( iterations ):
		currentP = e_step( X, MU, SIGMA, posterior )
		qp = quantisizeCurrentP( currentP )
		MU, SIGMA, posterior = m_step( X, qp )

	plot( X, MU, SIGMA, qp )

def plot( X, MU, SIGMA, currentP ):
	circle = []
	c = ['b','r']
	for i in range(2):
		circle.append(plt.Circle((MU[i][0],MU[i][1]), np.sqrt(SIGMA[i]), color=c[i], fill=False))

	ax = plt.gca()
	ax.add_artist(circle[0])
	ax.add_artist(circle[1])

	plt.scatter( X[:,0], X[:,1], c = currentP[1,:] )
	plt.show()

def readFile( src ):
	X = []
	with open( src ) as file:
		csvReader = csv.reader( file, delimiter=' ')
		for line in csvReader:
			X.append( [float(x) for x in line] )
	return np.array(X)

def initMu( X ):
	k = 2
	d = 2
	N = X.shape[0]
	MU = np.zeros([k, d])
	r1 = 0; r2 = 0
	while r1 == r2:
		r1 = np.random.randint( N )
		r2 = np.random.randint( N )
	MU[0] = X[r1,:]
	MU[1] = X[r2,:]
	return MU

def initSigma( X, MU ):
	d = 2
	sigmaSum = 0
	N = X.shape[0]
	for x, mu in zip(X, MU):
		sigmaSum += np.power( np.linalg.norm( x - mu ), 2 )
	sigma = 1 / (d * N) * sigmaSum
	return np.array([sigma, sigma])

# Hardcoded because I'm lazy
def initPosterior():
	k = 2
	prob = 1 / k
	posterior = np.array([prob, prob])
	return posterior

# Returns the probability of x belonging to this k:
# P(x|mu, sigma^2)
def p( x, mu, sigma ):
	d = 2
	N = 1 / np.power( (2 * np.pi * sigma), d / 2 ) * \
		np.exp( -1 / ( 2 * sigma ) * np.power( np.linalg.norm( x - mu ), 2 ) ) 
	return N

# Returns the probability for one point beloning to one cluster
def currentProbability( x, MU, SIGMA, posterior, i, t ):
	k = 2
	# P(x|theta)
	s = 0
	for j in range(k):
		s += posterior[j] * p( x, MU[j], SIGMA[j] )
	currentP = posterior[i] * p( x, MU[i], SIGMA[i] ) / s
	return currentP

def e_step( X, MU, SIGMA, posterior ):
	k = 2
	N = X.shape[0]
	currentP = np.zeros([k, N])
	for i in range(k):
		for t, x in enumerate(X):
			currentP[i, t] = currentProbability( x, MU, SIGMA, posterior, i, t )
	return currentP

def quantisizeCurrentP( currentP ):
	newP = np.zeros( currentP.shape )
	for i, c in enumerate(currentP[0,:]):
		if c < 0.5:
			newP[1,i] = 1
		else: 
			newP[0,i] = 1
	return newP

def m_step( X, currentP ):
	k = 2
	d = 2.0
	N = X.shape[0]
	n = [0] * k
	posterior = np.zeros(k)
	for i in range(k):
		for probability in currentP[i,:]:
			n[i] += probability
	# Getting mu
	MU = np.zeros([k, k])
	for i in range(k):
		posterior[i] = n[i] / N
		muSum = np.array([0.0,0.0])
		for probability, x in zip( currentP[i,:], X ):
			muSum += probability * x
		MU[i] = 1 / n[i] * muSum

	# Getting sigma
	SIGMA = np.zeros(k)
	for i in range(k):
		sigmaSum = 0
		for probability, x in zip( currentP[i,:], X ):	
			sigmaSum += probability * np.power( np.linalg.norm( x - MU[i] ), 2 )
		SIGMA[i] = 1 / (d * n[i]) * sigmaSum

	return MU, SIGMA, posterior

if __name__ == '__main__':
	main()
