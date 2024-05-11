# 定义逻辑回归模型
import numpy as np


# Sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 二元交叉熵损失函数
def binary_cross_entropy_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=10000): # 数据量小时可以将学习率调大或者迭代次数降低，这个主打一个稳
        self.lr = lr
        self.num_iter = num_iter
    
    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1])
        self.loss_history = []
        
        for _ in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            
            # 计算损失并存储历史记录
            loss = binary_cross_entropy_loss(y, h)
            self.loss_history.append(loss)
    
    def predict(self, X):
        return np.round(sigmoid(np.dot(X, self.theta)))
    


class Perceptron:
    def __init__(self, learning_rate=0.001, num_iter=1000):
        self.learning_rate = learning_rate
        self.num_iter = num_iter
    
    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        self.loss_history = []
        
        for _ in range(self.num_iter):
            # 计算激活函数值
            activation = np.dot(X, self.weights) + self.bias
            # 预测类别
            y_pred = np.where(activation >= 0, 1, 0)
            # 更新权重和偏置
            self.weights += self.learning_rate * np.dot(X.T, (y - y_pred))
            self.bias += self.learning_rate * np.sum(y - y_pred)
            # 计算损失并存储历史记录
            loss = np.mean(np.abs(y - y_pred))
            self.loss_history.append(loss)
    
    def predict(self, X):
        return np.where(np.dot(X, self.weights) + self.bias >= 0, 1, 0)
    
