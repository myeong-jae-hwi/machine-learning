import numpy as np
import matplotlib.pyplot as plt

def read_data(file_name):
    data = np.loadtxt(file_name)
    return data

def sigmoid(num):
    return 1 / (1 + np.exp(-num))

def log_loss(y, prediction):
    return -np.mean(y * np.log(prediction) + (1 - y) * np.log(1 - prediction))

def train_logistic_regression(X_train, y_train, X_valid, y_valid, epochs, learning_rate):
    a, b, c, d = np.random.rand(4)
    train_losses = []
    valid_losses = []
    
    for epoch in range(epochs):
        for i in range(len(X_train)):
            # 훈련 데이터에서 데이터 포인트 선택
            X_point = X_train[i]
            y_point = y_train[i]
            
            # 해당 데이터 포인트에 대한 예측
            z_train = a * X_point[0] + b * X_point[1] + c * X_point[2] + d
            prediction_train = sigmoid(z_train)

            z_valid = a * X_valid[i, 0] + b * X_valid[i, 1] + c * X_valid[i, 2] + d
            prediction_valid = sigmoid(z_valid)
            
            # 손실 계산
            loss_train = log_loss(y_point, prediction_train)
            train_losses.append(loss_train)
            
            loss_valid = log_loss(y_valid, prediction_valid)
            valid_losses.append(loss_valid)

            # 그래디언트 계산
            gradient_a = (prediction_train - y_point) * X_point[0]
            gradient_b = (prediction_train - y_point) * X_point[1]
            gradient_c = (prediction_train - y_point) * X_point[2]
            gradient_d = prediction_train - y_point
            
            # 매개변수 업데이트
            a -= learning_rate * gradient_a
            b -= learning_rate * gradient_b
            c -= learning_rate * gradient_c
            d -= learning_rate * gradient_d
                        
    return a, b, c, d, train_losses, valid_losses

# 데이터 읽기
testdata = read_data("test2.txt")
validdata = read_data("valid2.txt")
X_train = testdata[:, :3]
y_train = testdata[:, 3]

X_valid = validdata[:, :3]
y_valid = validdata[:, 3]

# 학습 파라미터 설정
epochs = 100  # 온라인 학습에서는 1 에포크로 설정
learning_rate = 0.1

# 학습 및 매개변수 추출
a, b, c, d, train_losses, valid_losses = train_logistic_regression(X_train, y_train, X_valid, y_valid, epochs, learning_rate)

# 손실 그래프 시각화
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Valid Loss')
plt.title('Loss Graph')
plt.show()
