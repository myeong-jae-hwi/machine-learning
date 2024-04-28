# import numpy as np
# import matplotlib.pyplot as plt

# # 파일에서 데이터 읽어오기 함수
# def read_data(file_name):
#     data = np.loadtxt(file_name)
#     return data

# # 데이터 파일 경로
# name = ["valid3.txt"]

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
    # 매개변수 초기화
    a, b, c, d = np.random.rand(4)
    best_acc = 0
    best_xi = None
    acc_arr = []
    
    for xi in xi_values:
        for epoch in range(epochs):
            # 모델 
            z = a * X[:, 0] + b * X[:, 1] + c * X[:, 2] + d
            prediction = sigmoid(z)
            
            # 그래디언트 계산
            gradient_a = np.mean((prediction - y) * X[:, 0])
            gradient_b = np.mean((prediction - y) * X[:, 1])
            gradient_c = np.mean((prediction - y) * X[:, 2])
            gradient_d = np.mean(prediction - y)
            
            # 매개변수 업데이트
            a -= learning_rate * gradient_a
            b -= learning_rate * gradient_b
            c -= learning_rate * gradient_c
            d -= learning_rate * gradient_d
            
        # 정확도 계산
        predictions = prediction >= 0.65
        acc = round(np.mean(predictions == y),3)
        acc_arr.append((xi, acc))
        
        if acc > best_acc:
            best_acc = acc
            best_xi = xi

    TN = np.sum((y == 0) & (predictions == 0))
    TP = np.sum((y == 1) & (predictions == 1))
    FN = np.sum((y == 1) & (predictions == 0))
    FP = np.sum((y == 0) & (predictions == 1))

    # TP = 파란색이라고 예측했는데 파란색
    # TN = 빨간색이라고 예측했는데 빨간색
    # FP = 파란색이라고 예측했는데 빨간색 (경계 위에 있는 빨간색)
    # FN = 빨간색이라고 예측했는데 파란색 (경계 아래에 있는 파란색)
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    print(f"2 -3 : 혼동행렬을 그린 뒤 각각의 값을 나타내라. recall :{round(TP / (TP + FN),2)}, acc :{round(((TP + TN) / 600),2)}, precision :{round((TP/(TP+FP)),2)}")
    return best_xi, best_acc, acc_arr, a, b, c, d

# 파일에서 데이터 읽기
testdata = read_data("test2.txt")
validdata = read_data("valid2.txt")
sumdata = read_data("test2.txt")
X = testdata[:, :3]
y = testdata[:, 3]

# 학습 파라미터 설정
epochs = 5000
learning_rate = 0.01
xi_values = np.linspace(0.1, 1, 10)

best_xi, best_acc, acc_arr, a, b, c, d = train_logistic_regression(X, y, epochs, learning_rate, xi_values)

X_valid = validdata[:, :3]
y_valid = validdata[:, 3]

# 검증 데이터셋에 대한 예측 수행
z_valid = a * X_valid[:, 0] + b * X_valid[:, 1] + c * X_valid[:, 2] + d
predictions_valid = sigmoid(z_valid) >= best_xi  # best_xi는 최적의 임계값

# 검증 데이터셋에 대한 평가 수행
accuracy_valid = np.mean(predictions_valid == y_valid)
print(f"Validation Set Accuracy: {accuracy_valid}")

print(f"Best ξ: {best_xi}, Best Accuracy: {best_acc}")
print(a, b, c, d)
for xi, acc in acc_arr:
    print(f"ξ: {xi}, Accuracy: {acc}")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 클래스에 따른 색상 지정
color_map = {0: 'r', 1: 'b'}

# 데이터셋에서 데이터 읽기 및 그리기
x, y, z, label = testdata[:, 0], testdata[:, 1], testdata[:, 2], testdata[:, 3]
colors = [color_map[int(i)] for i in label]
ax.scatter(x, y, z, c=colors, s=10)

# 결정 경계 그리기
x = np.linspace(min(x), max(x), 10)
y = np.linspace(min(y), max(y), 10)
X, Y = np.meshgrid(x, y)
Z = (-a/c)*X - (b/c)*Y - (d/c)

ax.plot_surface(X, Y, Z, alpha=0.5, color='black')

# 그래프 출력
plt.show()

# =-------------------------------------------------========================
