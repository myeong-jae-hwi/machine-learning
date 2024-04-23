# import numpy as np
# import matplotlib.pyplot as plt

# # 파일에서 데이터 읽어오기 함수
# def read_data(file_name):
#     data = np.loadtxt(file_name)
#     return data

# # 데이터 파일 경로
# name = ["train2.txt"]

# # 3차원 산점도 그리기
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # 클래스에 따른 색상 지정
# color_map = {0: 'r', 1: 'b'}

# # 각 파일로부터 데이터 읽어와서 그리기
# for file_name in name:
#     dataset = read_data(file_name)
#     x, y, z, label = dataset[:, 0], dataset[:, 1], dataset[:, 2], dataset[:, 3]
#     colors = [color_map[int(i)] for i in label]
#     ax.scatter(x, y, z, c=colors, s=10)

# # 그래프 출력
# plt.show()

# ==================================================================
# import numpy as np
# import matplotlib.pyplot as plt

# def read_data(file_name):
#     data = np.loadtxt(file_name)
#     return data

# def sigmoid(num):
#     return 1 / (1 + np.exp(-num))

# def log_loss(y, prediction):
#     return -np.mean(y * np.log(prediction) + (1 - y) * np.log(1 - prediction))

# def train_logistic_regression(X, y, epochs, learning_rate, xi_values):
#     # 매개변수 초기화
#     a, b, c, d = np.random.rand(4)
#     best_acc = 0
#     best_xi = None
#     acc_arr = []
    
#     for xi in xi_values:
#         for epoch in range(epochs):
#             # 모델 예측
#             z = a * X[:, 0] + b * X[:, 1] + c * X[:, 2] + d
#             prediction = sigmoid(z)
            
#             # 그래디언트 계산
#             gradient_a = np.mean((prediction - y) * X[:, 0])
#             gradient_b = np.mean((prediction - y) * X[:, 1])
#             gradient_c = np.mean((prediction - y) * X[:, 2])
#             gradient_d = np.mean(prediction - y)
            
#             # 매개변수 업데이트
#             a -= learning_rate * gradient_a
#             b -= learning_rate * gradient_b
#             c -= learning_rate * gradient_c
#             d -= learning_rate * gradient_d
            
#         # 정확도 계산
#         predictions = prediction >= 0.5
#         acc = round(np.mean(predictions == y),3)
#         acc_arr.append((xi, acc))
        
#         if acc > best_acc:
#             best_acc = acc
#             best_xi = xi
    
#     return best_xi, best_acc, acc_arr, a, b, c, d

# # 파일에서 데이터 읽기
# dataset = read_data("valid2.txt")
# X = dataset[:, :3]
# y = dataset[:, 3]

# # 학습 파라미터 설정
# epochs = 1000
# learning_rate = 0.01
# xi_values = np.linspace(0.01, 1, 100)

# best_xi, best_acc, acc_arr, a, b, c, d = train_logistic_regression(X, y, epochs, learning_rate, xi_values)

# print(f"Best ξ: {best_xi}, Best Accuracy: {best_acc}")
# print(a, b, c, d)
# for xi, acc in acc_arr:
#     print(f"ξ: {xi}, Accuracy: {acc}")

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # 클래스에 따른 색상 지정
# color_map = {0: 'r', 1: 'b'}

# # 데이터셋에서 데이터 읽기 및 그리기
# x, y, z, label = dataset[:, 0], dataset[:, 1], dataset[:, 2], dataset[:, 3]
# colors = [color_map[int(i)] for i in label]
# ax.scatter(x, y, z, c=colors, s=10)

# # 결정 경계 그리기
# x = np.linspace(min(x), max(x), 10)
# y = np.linspace(min(y), max(y), 10)
# X, Y = np.meshgrid(x, y)
# Z = (-a/c)*X - (b/c)*Y - (d/c)

# ax.plot_surface(X, Y, Z, alpha=0.5, color='y', edgecolors='w')

# # 그래프 출력
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

def read_data(file_name):
    data = np.loadtxt(file_name)
    return data

def sigmoid(num):
    return 1 / (1 + np.exp(-num))

def log_loss(y, prediction):
    return -np.mean(y * np.log(prediction) + (1 - y) * np.log(1 - prediction))

def train_logistic_regression(X, y, epochs, learning_rate, xi_values):
    a, b, c, d = np.random.rand(4)
    best_acc = 0
    best_xi = None
    acc_arr = []
    
    for xi in xi_values:
        for epoch in range(epochs):
            z = a * X[:, 0] + b * X[:, 1] + c * X[:, 2] + d
            prediction = sigmoid(z)
            
            gradient_a = np.mean((prediction - y) * X[:, 0])
            gradient_b = np.mean((prediction - y) * X[:, 1])
            gradient_c = np.mean((prediction - y) * X[:, 2])
            gradient_d = np.mean(prediction - y)
            
            a -= learning_rate * gradient_a
            b -= learning_rate * gradient_b
            c -= learning_rate * gradient_c
            d -= learning_rate * gradient_d
            
        predictions = prediction >= 0.5
        acc = round(np.mean(predictions == y),3)
        acc_arr.append((xi, acc))
        
        if acc > best_acc:
            best_acc = acc
            best_xi = xi
    
    return best_xi, best_acc, acc_arr, a, b, c, d

def confusion_matrix_visualization(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    confusion_matrix = np.array([[TP, FP], [FN, TN]])
    
    fig, ax = plt.subplots()
    cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + ['P', 'N'])
    ax.set_yticklabels([''] + ['T', 'F'])
    
    for (i, j), val in np.ndenumerate(confusion_matrix):
        ax.text(j, i, f'{val}', ha='center', va='center', color='red')
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# 데이터 로드 및 모델 학습
dataset = read_data("train2.txt")
X = dataset[:, :3]
y = dataset[:, 3]

epochs = 1000
learning_rate = 0.01
xi_values = np.linspace(0.01, 1, 100)

best_xi, best_acc, acc_arr, a, b, c, d = train_logistic_regression(X, y, epochs, learning_rate, xi_values)

# 모델을 사용하여 예측
z = a * X[:, 0] + b * X[:, 1] + c * X[:, 2] + d
predictions = sigmoid(z) >= 0.5

# 혼동 행렬 시각화
confusion_matrix_visualization(y, predictions)
