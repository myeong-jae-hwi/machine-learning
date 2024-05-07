# import numpy as np
# import matplotlib.pyplot as plt

# # 파일에서 데이터 읽어오기 함수
# def read_data(file_name):
#     data = np.loadtxt(file_name)
#     return data

# # 데이터 파일 경로
# name = ["test2.txt","valid2.txt","train2.txt"]

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
import warnings
warnings.filterwarnings('ignore')

def read_data(file_name):
    data = np.loadtxt(file_name)
    return data

def sigmoid(num):
    return 1 / (1 + np.exp(-num))

def log_loss(y, prediction):
    return -np.mean(y * np.log(prediction) + (1 - y) * np.log(1 - prediction))


def train_logistic_regression(X, y, epochs, learning_rate,step_values, validX, validY):
    # 매개변수 초기화
    a, b, c, d = np.random.rand(4)
    best_acc = 0
    beststep = None
    acc_arr = []
    acc_arr_test = []  # 훈련 손실 값 저장 리스트
    acc_arr_valid = []  # 검증 손실 값 저장 리스트
    abcd_arr = []
    
    for step in step_values:
        for epoch in range(epochs):
            # 모델 
            z = a * X[:, 0] + b * X[:, 1] + c * X[:, 2] + d
            prediction = sigmoid(z)

            z_valid = a * validX[:, 0] + b * validX[:, 1] + c * validX[:, 2] + d
            prediction_valid = sigmoid(z_valid)

            loss_valid = log_loss(validY, prediction_valid)
            
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

            predictions_valid = sigmoid(z_valid) >= 0.65

            # acc 구하는 부분
            predictions = prediction >= 0.65
            acc = round(np.mean(predictions == y),3)
            acc_arr.append((step, acc))

            # print(f"test acc : {acc}")
            # print(f"test Loss: {round((1 - acc),2)}")

            accuracy_valid = round(np.mean(predictions_valid == validY),2)
            abcd_arr.append((a, b, c, d))  # 현재 파라미터 저장
            acc_arr_valid.append((1 - accuracy_valid))
            acc_arr_test.append((1 - acc))
    
        print(f"파라미터 업데이트 과정 :[{step}] ",round(a,2), round(b,2), round(c,2), round(d,2))
        z_valid = a * validX[:, 0] + b * validX[:, 1] + c * validX[:, 2] + d

        if acc > best_acc:
            best_acc = acc
            beststep = step

    TN = np.sum((y == 0) & (predictions == 0))
    TP = np.sum((y == 1) & (predictions == 1))
    FN = np.sum((y == 1) & (predictions == 0))
    FP = np.sum((y == 0) & (predictions == 1))

    # TP = 파란색이라고 예측했는데 파란색
    # TN = 빨간색이라고 예측했는데 빨간색
    # FP = 파란색이라고 예측했는데 빨간색 (경계 위에 있는 빨간색)
    # FN = 빨간색이라고 예측했는데 파란색 (경계 아래에 있는 파란색)
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    print(f"2 -3 : 혼동행렬을 그린 뒤 각각의 값을 나타내라. recall :{round(TP / (TP + FN),2)}, acc :{((TP + TN) / 600)}, precision :{round((TP/(TP+FP)),2)}")
    print(f"Validation Loss: {loss_valid}")

    plt.figure(figsize=(12, 6))
    plt.plot(acc_arr_valid, label='test loss', color='red')
    plt.plot(acc_arr_test, label='valid loss',color='blue')

    for i in range(1, len(acc_arr_valid)):
        if acc_arr_valid[i] > acc_arr_valid[i-1]: 
            plt.axvline(x=i, color='green', linestyle='--')
            print(f"파라미터: a={round(abcd_arr[i][0],2)}, b={round(abcd_arr[i][1],2)}, c={round(abcd_arr[i][2],2)}, d={round(abcd_arr[i][3],2)}")

    plt.show()

    return beststep, best_acc, acc_arr, round(a,2), round(b,2), round(c,2), round(d,2)

# 파일에서 데이터 읽기
testdata = read_data("test2.txt")
validdata = read_data("valid2.txt")
X = testdata[:, :3]
y = testdata[:, 3]
validX = validdata[:, :3]
validY = validdata[:, 3]

# 학습 파라미터 설정
epochs = 500
learning_rate = 0.01 
step_values = np.linspace(0.1, 1, 10)

beststep, best_acc, acc_arr, a, b, c, d = train_logistic_regression(X, y, epochs, learning_rate,step_values, validX, validY)

print(f"Best ξ: {beststep}, Best Accuracy: {best_acc}")

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

# # =----------------------3번 디씨전트리---------------------------========================
