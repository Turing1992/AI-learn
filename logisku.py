from ML.Logis.logis_RX import MLPClassifier
from pylab import *

mpl.rcParams['font.sans-serif']=['SimHei']

train_data=np.loadtxt('train_car.txt',delimiter=',')
test_data=np.loadtxt('test_car.txt',delimiter=',')

train_X,train_y=train_data[:,0:2],train_data[:,-1]
test_X,test_y=test_data[:,0:2],test_data[:,-1]

model=MLPClassifier()
model.fit(train_X,train_y,alpha=0.01,max_iter=15000)
# print(model.J_history_)
# plt.plot(model.J_history_)
# plt.show()
# print(model.score(train_X,train_y))
# print(model.score(test_X,test_y))
# model.zero_one_map(test_X,test_y)

theta=model.theta_
print(theta)
x_min=np.min(train_X[:,1])