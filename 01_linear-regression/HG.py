import numpy as np
import matplotlib.pyplot as plt
import random

def showGraph(x, y, y_pred):
	plt.scatter(x,y)
	plt.plot([min(x), max(x)], [min(y_pred), max(y_pred)], color='red')
	plt.show()


def randomLinearFunctionGenerator(n=5, m=100):
	betas = np.array([np.random.randint(1,3)+np.random.random() for _ in range(n+1)])
	X = np.append(np.ones((m,1)), np.random.randint(-10,10,size=(m,n)),axis=1)
	y = np.matmul(X,betas)+np.random.randint(-5,5)
	return betas,X,y


def linearRegression(X, y, epochs = 1000, lr = 0.001):
	betas = np.zeros(X.shape[1])
	y_pred = 0
	m = len(y)
	for epoch in range(epochs):
		y_pred = np.matmul(X,betas)
		for ii, x in enumerate(X.transpose()):
			loss = -2*np.mean((y-y_pred)*x)
			betas[ii] -= lr*loss
	return betas


def main():
	n=5
	m=100
	betas,X,y = randomLinearFunctionGenerator(n,m)
	betas_pred = linearRegression(X,y)
	print("betas: {}".format(betas))
	print("betas_pred: {}".format(betas_pred))
	if n==1:
		showGraph(X.transpose()[1],y,np.matmul(X,betas_pred))


if __name__=='__main__':
	main()