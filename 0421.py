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
#         predictions = prediction >= 0.65
#         acc = round(np.mean(predictions == y),3)
#         acc_arr.append((xi, acc))
        
#         if acc > best_acc:
#             best_acc = acc
#             best_xi = xi

#     TN = np.sum((y == 0) & (predictions == 0))
#     TP = np.sum((y == 1) & (predictions == 1))
#     FN = np.sum((y == 1) & (predictions == 0))
#     FP = np.sum((y == 0) & (predictions == 1))

#     print(TP,TN,FP,FN)
#     print(f"2 -3 : 혼동행렬을 그린 뒤 각각의 값을 나타내라. recall :{TP / (TP + FN)}, acc :{(TP + TN) / 600}, precision :{TP/(TP+FP)}")
#     return best_xi, best_acc, acc_arr, a, b, c, d

# # 파일에서 데이터 읽기
# dataset = read_data("test2.txt")
# X = dataset[:, :3]
# y = dataset[:, 3]

# # 학습 파라미터 설정
# epochs = 5000
# learning_rate = 0.01
# xi_values = np.linspace(0.1, 1, 10)

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

# ax.plot_surface(X, Y, Z, alpha=0.5, color='black')

# # 그래프 출력
# plt.show()


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
#     error_arr = []  # 에러 값 기록
    
#     for xi in xi_values:
#         for epoch in range(epochs):
#             # 모델 예측
#             z = a * X[:, 0] + b * X[:, 1] + c * X[:, 2] + d
#             prediction = sigmoid(z)
            
#             # 로그 손실 계산 및 기록
#             error = log_loss(y, prediction)
#             error_arr.append(error)
            
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
#         predictions = prediction >= 0.65
#         acc = round(np.mean(predictions == y), 3)
#         acc_arr.append((xi, acc))
        
#         if acc > best_acc:
#             best_acc = acc
#             best_xi = xi
    
#     # 그래프 출력
#     plt.plot(error_arr)
#     plt.xlabel('Epoch')
#     plt.ylabel('Error')
#     plt.title('Error over Epochs')
#     plt.xlim(0, epochs)  # x 축 범위 설정

#     plt.show()
    
#     return best_xi, best_acc, acc_arr, a, b, c, d

# # 파일에서 데이터 읽기
# dataset = read_data("test2.txt")
# X = dataset[:, :3]
# y = dataset[:, 3]

# # 학습 파라미터 설정
# epochs = 1000
# learning_rate = 0.1
# xi_values = np.linspace(0.1, 1, 10)

# best_xi, best_acc, acc_arr, a, b, c, d = train_logistic_regression(X, y, epochs, learning_rate, xi_values)

# print(f"Best ξ: {best_xi}, Best Accuracy: {best_acc}")
# print(a, b, c, d)
# for xi, acc in acc_arr:
#     print(f"ξ: {xi}, Accuracy: {acc}")


# import numpy as np
# import matplotlib.pyplot as plt

# def read_data(file_name):
#     data = np.loadtxt(file_name)
#     return data

# def sigmoid(num):
#     return 1 / (1 + np.exp(-num))

# def log_loss(y, prediction):
#     return -np.mean(y * np.log(prediction) + (1 - y) * np.log(1 - prediction))

# def train_logistic_regression(X_train, y_train, X_valid, y_valid, epochs, learning_rate, xi_values):
#     # 매개변수 초기화
#     a, b, c, d = np.random.rand(4)
#     best_acc = 0
#     best_xi = None
#     acc_arr = []
#     error_arr_train = []  # 훈련 데이터 에러 값 기록
#     error_arr_valid = []  # 검증 데이터 에러 값 기록
    
#     for xi in xi_values:
#         for epoch in range(epochs):
#             # 훈련 데이터로 모델 예측 및 업데이트
#             z_train = a * X_train[:, 0] + b * X_train[:, 1] + c * X_train[:, 2] + d
#             prediction_train = sigmoid(z_train)
            
#             # 로그 손실 계산 및 기록
#             error_train = log_loss(y_train, prediction_train)
#             error_arr_train.append(error_train)
            
#             # 그래디언트 계산
#             gradient_a = np.mean((prediction_train - y_train) * X_train[:, 0])
#             gradient_b = np.mean((prediction_train - y_train) * X_train[:, 1])
#             gradient_c = np.mean((prediction_train - y_train) * X_train[:, 2])
#             gradient_d = np.mean(prediction_train - y_train)
            
#             # 매개변수 업데이트
#             a -= learning_rate * gradient_a
#             b -= learning_rate * gradient_b
#             c -= learning_rate * gradient_c
#             d -= learning_rate * gradient_d
            
#             # 검증 데이터로 모델 예측
#             z_valid = a * X_valid[:, 0] + b * X_valid[:, 1] + c * X_valid[:, 2] + d
#             prediction_valid = sigmoid(z_valid)
            
#             # 검증 데이터 에러 계산
#             error_valid = log_loss(y_valid, prediction_valid)
#             error_arr_valid.append(error_valid)
            
#         # 정확도 계산
#         predictions = prediction_valid >= 0.65
#         acc = round(np.mean(predictions == y_valid), 3)
#         acc_arr.append((xi, acc))
        
#         if acc > best_acc:
#             best_acc = acc
#             best_xi = xi
    
#     # 그래프 출력
#     plt.plot(error_arr_train, label='Training Data')
#     plt.plot(error_arr_valid, label='Validation Data')
#     plt.xlabel('Epoch')
#     plt.ylabel('Error')
#     plt.title('Error over Epochs')
#     plt.legend()
#     plt.xlim(0, epochs)  # x 축 범위 설정

#     plt.show()
    
#     return best_xi, best_acc, acc_arr, a, b, c, d

# # 파일에서 데이터 읽기
# train_data = read_data("test2.txt")
# valid_data = read_data("valid2.txt")

# X_train = train_data[:, :3]
# y_train = train_data[:, 3]

# X_valid = valid_data[:, :3]
# y_valid = valid_data[:, 3]

# # 학습 파라미터 설정
# epochs = 1000
# learning_rate = 0.1
# xi_values = np.linspace(0.1, 1, 10)

# best_xi, best_acc, acc_arr, a, b, c, d = train_logistic_regression(X_train, y_train, X_valid, y_valid, epochs, learning_rate, xi_values)

# print(f"Best ξ: {best_xi}, Best Accuracy: {best_acc}")
# print(a, b, c, d)
# for xi, acc in acc_arr:
#     print(f"ξ: {xi}, Accuracy: {acc}")

import numpy as np
import matplotlib.pyplot as plt

def read_data(file_name):
    data = np.loadtxt(file_name)
    return data

def sigmoid(num):
    return 1 / (1 + np.exp(-num))

def log_loss(y, prediction):
    return -np.mean(y * np.log(prediction) + (1 - y) * np.log(1 - prediction))

def train_logistic_regression(X_train, y_train, X_valid, y_valid, epochs, learning_rate, xi_values):
    # 매개변수 초기화
    a, b, c, d = np.random.rand(4)
    best_acc = 0
    best_xi = None
    acc_arr = []
    error_arr_train = []  # 훈련 데이터 에러 값 기록
    error_arr_valid = []  # 검증 데이터 에러 값 기록
    
    for xi in xi_values:
        for epoch in range(epochs):
            # 훈련 데이터로 모델 예측 및 업데이트
            z_train = a * X_train[:, 0] + b * X_train[:, 1] + c * X_train[:, 2] + d
            prediction_train = sigmoid(z_train)
            
            # 로그 손실 계산 및 기록
            error_train = log_loss(y_train, prediction_train)
            error_arr_train.append(error_train)
            
            # 그래디언트 계산
            gradient_a = np.mean((prediction_train - y_train) * X_train[:, 0])
            gradient_b = np.mean((prediction_train - y_train) * X_train[:, 1])
            gradient_c = np.mean((prediction_train - y_train) * X_train[:, 2])
            gradient_d = np.mean(prediction_train - y_train)
            
            # 매개변수 업데이트
            a -= learning_rate * gradient_a
            b -= learning_rate * gradient_b
            c -= learning_rate * gradient_c
            d -= learning_rate * gradient_d
            
            # 검증 데이터로 모델 예측
            z_valid = a * X_valid[:, 0] + b * X_valid[:, 1] + c * X_valid[:, 2] + d
            prediction_valid = sigmoid(z_valid)
            
            # 검증 데이터 에러 계산
            error_valid = log_loss(y_valid, prediction_valid)
            error_arr_valid.append(error_valid)
            
        # 정확도 계산
        predictions = prediction_valid >= 0.65
        acc = round(np.mean(predictions == y_valid), 3)
        acc_arr.append((xi, acc))
        
        if acc > best_acc:
            best_acc = acc
            best_xi = xi
    
    # 그래프 출력
    plt.plot(np.log(error_arr_train), label='Training Data')
    plt.plot(np.log(error_arr_valid), label='Validation Data')
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.title('Log Loss over Epochs')
    plt.xlim(0, epochs)  # x 축 범위 설정

    plt.legend()
    plt.show()
    
    return best_xi, best_acc, acc_arr, a, b, c, d

# 파일에서 데이터 읽기
train_data = read_data("test2.txt")
valid_data = read_data("valid2.txt")

X_train = train_data[:, :3]
y_train = train_data[:, 3]

X_valid = valid_data[:, :3]
y_valid = valid_data[:, 3]

# 학습 파라미터 설정
epochs = 1000
learning_rate = 0.1
xi_values = np.linspace(0.1, 1, 10)

best_xi, best_acc, acc_arr, a, b, c, d = train_logistic_regression(X_train, y_train, X_valid, y_valid, epochs, learning_rate, xi_values)

print(f"Best ξ: {best_xi}, Best Accuracy: {best_acc}")
print(a, b, c, d)
for xi, acc in acc_arr:
    print(f"ξ: {xi}, Accuracy: {acc}")
