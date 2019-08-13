import numpy as np
import matplotlib.pyplot as plt

class MLPClassifier:
    def __init__(self):
        self.theta_=None
        self.J_history_=None

    def featureprocessing(self,X,y):
        mu=np.mean(X,0)
        sigma=np.std(X,0)
        X-=mu
        X/=sigma
        X=np.c_[np.ones(X.shape[0]),X]
        y=np.c_[y]
        return X,y

    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))

    def fit(self,X,y,alpha,max_iter,epsion=1e-4):
        def costFunction(h,y,m):
            return (-1.0/m)*(np.sum(y*np.log(h)+(1.0-y)*np.log(1-h)))
        X,y=self.featureprocessing(X,y)
        m,f=X.shape
        theta=np.zeros((f,1))
        J_history=np.zeros(max_iter)
        min_iter=0
        while min_iter<max_iter:
            h=self.sigmoid(np.dot(X,theta))
            J_history[min_iter]=costFunction(h,y,m)
            deltatheta=(1.0/m)*X.T.dot(h-y)
            theta-=alpha*deltatheta
            min_iter+=1
        self.theta_=theta
        self.J_history_=J_history

    def prediect(self,X,y):
        X,y=self.featureprocessing(X,y)
        pred_y=self.sigmoid(np.dot(X,self.theta_))
        return np.where(pred_y>0.5,1,0)

    def score(self,test_X,test_y):
        count=0
        test_X,test_y=self.featureprocessing(test_X,test_y)
        pred_y=self.sigmoid(np.dot(test_X,self.theta_))
        for i in range(len(pred_y)):
            pred_y=np.where(pred_y>0.5,1,0)
            if pred_y[i]==test_y[i]:
                count+=1
        return count/len(test_X)

    def zero_one_map(self,X,y):
        X,y=self.featureprocessing(X,y)
        for i in range(len(X)):
            if y[i]==1:
                plt.scatter(X[i,1],X[i,2],c='r',marker='x')
            else:
                plt.scatter(X[i,1],X[i,2],c='b',marker='x')
        plt.show()