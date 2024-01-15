# @Author  : Gong Zheng
# @Function:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 损失函数
def function_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost = cost + (f_wb_i - y[i])**2
    cost = cost / (2 * m)
    return cost


# 计算偏导数
def function_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros((n,))  # 一行n列
    dj_db = 0.

    for i in range(m):
        error = np.dot(X[i], w) + b - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + error * X[i, j]
        dj_db = dj_db + error

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def gradient_descent(X, y, w_star, b_star, alpha, iterations, cost_function, gradient_function):
    # 如果x为空，结束
    m = X.shape[0]
    if m == 0:
        return 0

    j_history = []
    w = w_star
    b = b_star

    for i in range(iterations):
        cost = cost_function(X, y, w, b)
        dj_w, dj_b = gradient_function(X, y, w, b)

        j_history.append(cost)
        w = w - alpha * dj_w
        b = b - alpha * dj_b

    return w, b, j_history

# 归一化
def Zscore_Normalization(X):
    x_mean = np.mean(X, axis=0)
    x_sigma = np.std(X, axis=0)
    x_norm = (X - x_mean) / x_sigma

    return x_norm


if __name__ == '__main__':
    X_train = np.genfromtxt("../data/housing.csv", usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
    y_train = np.genfromtxt("../data/housing.csv", usecols=(13))

    X_norm = Zscore_Normalization(X_train)

    w_in = np.zeros(X_train.shape[1])
    b_in = 0.
    iterations = 1000
    alpha = 3e-1

    w_final, b_final, j_hist = gradient_descent(X_norm, y_train, w_in, b_in, alpha, iterations,
                                                function_cost, function_gradient)
    print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")

    # 绘图
    iteration = np.linspace(0, iterations, iterations)
    plt.subplot(1, 2, 1)
    plt.plot(iteration[0:100], j_hist[0:100], color='y')
    plt.title("0:100")
    plt.xlabel("iteration")
    plt.ylabel("cost")
    plt.subplot(1, 2, 2)
    plt.plot(iteration[500:1000], j_hist[500:1000], color='y')
    plt.title("500:1000")
    plt.xlabel("iteration")
    plt.ylabel("cost")
    plt.show()

    y_pre = np.dot(X_norm, w_final) + b_final
    plt.scatter(X_train.T[0], y_pre, edgecolors='yellow')
    plt.scatter(X_train.T[0],y_train, edgecolors='blue')
    plt.show()