from model import *
import matplotlib.pyplot as plt
# def main():
#         # 读取数据
#     data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)

#     # 提取特征和标签
#     X = data[:, :-1]
#     y = data[:, -1]

#     # 数据标准化
#     X_mean = np.mean(X, axis=0)
#     X_std = np.std(X, axis=0)
#     X_scaled = (X - X_mean) / X_std

#     # 添加偏置项
#     X_scaled = np.column_stack((np.ones(X_scaled.shape[0]), X_scaled))

#     # 分割数据集
#     np.random.seed(42)
#     indices = np.random.permutation(X_scaled.shape[0])
#     # 80% 的数据被用作训练集，而剩余的20% 被用作测试集
#     train_indices, test_indices = indices[:int(0.8*X_scaled.shape[0])], indices[int(0.8*X_scaled.shape[0]):]
#     X_train, X_test = X_scaled[train_indices], X_scaled[test_indices]
#     y_train, y_test = y[train_indices], y[test_indices]

#     # 训练逻辑回归模型
#     lr_model = LogisticRegression()
#     lr_model.fit(X_train, y_train)
#     lr_pred = lr_model.predict(X_test)

#     # 计算模型准确率
#     accuracy = np.mean(lr_pred == y_test)
#     print("Logistic Regression Accuracy:", accuracy)

#     # 获取模型参数
#     theta0 = lr_model.theta[0]
#     theta1 = lr_model.theta[1]
#     theta2 = lr_model.theta[2]

#     # 绘制数据点
#     plt.figure(figsize=(10, 6))
#     plt.scatter(X_test[:, 1], X_test[:, 2], c=y_test, cmap='coolwarm', marker='o', label='Actual')

#     # 绘制逻辑回归模型预测结果
#     plt.scatter(X_test[:, 1], X_test[:, 2], c=lr_pred, cmap='coolwarm', marker='o', label='Logistic Regression Prediction')

#     # 绘制决策边界
#     x_values = np.linspace(-2, 2, 10)
#     y_values = -(theta1 * x_values + theta0) / theta2
#     plt.plot(x_values, y_values, color='red', linestyle='solid', linewidth=2, label='Decision Boundary')

#     plt.xlabel('Age (Scaled)')
#     plt.ylabel('Estimated Salary (Scaled)')
#     plt.title('Logistic Regression Predictions vs Actual')
#     plt.legend()
#     plt.show()

#     # 绘制 loss 曲线图
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(len(lr_model.loss_history)), lr_model.loss_history)
#     plt.xlabel('Iteration')
#     plt.ylabel('Binary Cross-Entropy Loss')
#     plt.title('Training Loss Curve')
#     plt.show()



def main():
    # 读取数据
    data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)

    # 提取特征和标签
    X = data[:, :-1]
    y = data[:, -1]

    # 数据标准化
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_scaled = (X - X_mean) / X_std

    # 添加偏置项
    X_scaled = np.column_stack((np.ones(X_scaled.shape[0]), X_scaled))

    # 分割数据集
    np.random.seed(81)
    indices = np.random.permutation(X_scaled.shape[0])
    # 80% 的数据被用作训练集，而剩余的20% 被用作测试集
    train_indices, test_indices = indices[:int(0.59*X_scaled.shape[0])], indices[int(0.59*X_scaled.shape[0]):]
    X_train, X_test = X_scaled[train_indices], X_scaled[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    # 训练感知机模型
    perceptron_model = Perceptron()
    perceptron_model.fit(X_train, y_train)
    perceptron_pred = perceptron_model.predict(X_test)

    # 计算模型准确率
    accuracy = np.mean(perceptron_pred == y_test)
    print("Perceptron Accuracy:", accuracy)

    # 绘制数据点
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test[:, 1], X_test[:, 2], c=y_test, cmap='coolwarm', marker='o', label='Actual')

    # 绘制感知机模型预测结果
    plt.scatter(X_test[:, 1], X_test[:, 2], c=perceptron_pred, cmap='coolwarm', marker='o', label='Perceptron Prediction')

    # 绘制决策边界
    theta0 = perceptron_model.bias
    theta1, theta2 = perceptron_model.weights[1], perceptron_model.weights[2]
    x_values = np.linspace(-2, 2, 10)
    y_values = -(theta1 * x_values + theta0) / theta2
    plt.plot(x_values, y_values, color='red', linestyle='solid', linewidth=2, label='Decision Boundary')

    plt.xlabel('Age (Scaled)')
    plt.ylabel('Estimated Salary (Scaled)')
    plt.title('Perceptron Predictions vs Actual')
    plt.legend()
    plt.show()

    # 获取损失历史记录
    loss_history = perceptron_model.loss_history

    # 绘制损失曲线图
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(loss_history)), loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Number of Misclassifications')
    plt.title('Perceptron Loss Curve')
    plt.show()
    

if __name__ == '__main__':
    main()
