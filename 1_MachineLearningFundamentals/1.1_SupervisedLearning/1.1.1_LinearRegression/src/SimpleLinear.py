# @Author  : Gong Zheng
# @Function: 跑test数据
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 损失函数 J(w,b)
def cost_function(x, y, w, b):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i]) ** 2

    cost_total = cost / (2 * m)

    return cost_total

# 梯度下降法
def gradient_function(x, y, w, b):
    m = x.shape[0]

    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        diver = f_wb - y[i]
        dj_dw += diver * x[i]
        dj_db += diver
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def gradient_descent(x, y, w_star, b_star, alpha, iterations, cost_function, gradient_function):
    m = x.shape[0]
    if m == 0:
        return 0

    # 记录数据，方便回顾
    w_history = []
    b_history = []
    j_history = []

    w = w_star
    b = b_star

    for i in range(iterations):
        w_history.append(w)
        b_history.append(b)

        cost = cost_function(x, y, w, b)
        dj_w, dj_b = gradient_function(x, y, w, b)

        w = w - alpha * dj_w
        b = b - alpha * dj_b

        j_history.append(cost)

        if i % 10000 == 0:
            print(f"Iteration {i:4}: Cost {j_history[-1]:0.2e} ",
                  f"dj_dw: {dj_w: 0.3e}, dj_db: {dj_b: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w, b, j_history, w_history, b_history


if __name__ == '__main__':
    csv = pd.read_csv("../data/test.csv")
    x_train = np.array(csv.x)
    y_train = np.array(csv.y)

    plt.scatter(x_train, y_train)
    plt.show()

    w_star = 0
    b_star = 0

    iterations = 100000
    # 数据量大，需要选择更小学习率
    alpha = 1.0e-4

    w_final, b_final, j_hist, w_hist, b_hist = gradient_descent(x_train, y_train, w_star, b_star, alpha, iterations,
                                                                cost_function, gradient_function)

    print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

    # 观察收敛性
    iteration = np.linspace(0, iterations, iterations)
    plt.scatter(iteration[0:20], j_hist[0:20])
    plt.plot(iteration[0:20], j_hist[0:20], color='y')
    plt.show()

    # 模拟曲线
    f_x = np.linspace(x_train.min(), x_train.max(), len(x_train))
    f_y = w_final * f_x + b_final
    plt.scatter(x_train, y_train)
    plt.plot(f_x, f_y, color='y')
    plt.show()