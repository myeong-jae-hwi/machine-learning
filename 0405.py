import pandas as fubao
import numpy as np
import matplotlib.pyplot as plt
# # # ============================ 과제 1 ============================
data = fubao.read_csv('과제1_데이터.tsv',delimiter='\t')

a_class = data[data['label'] == 0]
b_class = data[data['label'] == 1]

plt.scatter(a_class['feat1'], a_class['feat2'], color='blue')
plt.scatter(b_class['feat1'], b_class['feat2'], color='orange')

plt.show()
# ================================================================

# ============================ 과제 2 ============================
import random

# data = fubao.read_csv('과제1_데이터.tsv',delimiter='\t')
#
# a_class = data[data['label'] == 0]
# b_class = data[data['label'] == 1]
#
# # # 초기값 설정
# a = random.randint(1, 10)
# b = random.randint(1, 10)
# lr = 0.5
# prev_loss = 0
# count = 0   # 같은 로스가 나온 횟수
# cnt = 0     # 최종 횟수
#
# # 로스가 20번동안 같을 때 까지 반복
# while count < 20:
#     # 모델 학습
#     model = a * data['feat1'] + b
#     predictions = (model >= data['feat2']).astype(int)    # 에측치
#     correct_predictions = (predictions == data['label']).sum()  # 예측치가 맞은 개수
#     loss = 1 - (correct_predictions / 200)
#
#     new_a = np.sum(data['feat1'] * (predictions - data['label'])) / 200   # 기울기 구하는 부분
#     new_b = np.sum(predictions - data['label']) / 200                     # y절편 구하는 부분
#
#     # 업데이트
#     a -= lr * new_a
#     b -= lr * new_b
#
#     # 로스가 이전과 같은지 확인
#     if loss == prev_loss:
#         count += 1
#         cnt += 1
#     else:
#         count = 0
#     prev_loss = loss
#
#     # 경계선
#     x_values = np.linspace(min(data['feat1']), max(data['feat1']), 2)
#     y_values = a * x_values + b
#
#     # 로스 출력
#     # print(correct_predictions)
#     # print("gradient_a:", a)
#     # print("gradient_b:", b)
#     # print("loss: ", round(loss,2))
#     # print(data['feat1'])
#
#
# # 그래프
# plt.scatter(a_class['feat1'], a_class['feat2'],s=10)
# plt.scatter(b_class['feat1'], b_class['feat2'], color='orange',s=10)
# plt.plot(x_values, y_values, color='red')
#
# plt.show()

# ============================ 과제 3 ============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
data = pd.read_csv('과제1_데이터.tsv', delimiter='\t')

# 클래스 분리
a_class = data[data['label'] == 0]
b_class = data[data['label'] == 1]

# 특성 추출
feat1 = data['feat1'].values
feat2 = data['feat2'].values

# 시그모이드 함수 정의
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 로지스틱 회귀 학습 및 시각화 함수 정의
def logistic_regression_and_plot(data, feat1, feat2, learning_rate, num_iterations):
    # 초기 파라미터 설정
    a = 1
    b = 2
    c = 3
    l = learning_rate

    # 손실 기록을 위한 리스트 초기화
    arr_z = []
    arr_hz = []
    arr_logloss = []

    # 경사 하강법을 통한 파라미터 업데이트
    for i in range(num_iterations):
        # z 값 계산
        z = (a * feat1) + (b * feat2) + c
        arr_z.append(z)

        # 시그모이드 함수를 통해 hz(예측값) 계산
        hz = sigmoid(z)
        arr_hz.append(hz)

        # 로그 손실 계산
        log_loss = -(l * np.log(hz + 1e-15) + (1 - l) * np.log(1 - hz + 1e-15))
        arr_logloss.append(log_loss)

        # 그래디언트 계산
        gradient_a = np.sum((hz - data['label']) * feat1)
        gradient_b = np.sum((hz - data['label']) * feat2)
        gradient_c = np.sum(hz - data['label'])

        # 파라미터 업데이트
        a -= l * gradient_a
        b -= l * gradient_b
        c -= l * gradient_c

        # 로그 손실 출력
        if (i+1) % 100 == 0:
            print("Iteration:", i+1, "Loss:", np.mean(log_loss))

    # 직선의 방정식을 ax1 + bx2 + c로 표현
    x1_value = np.linspace(min(data['feat1']), max(data['feat1']), 2)
    x2_value = (-a * x1_value - c) / b

    # 그래프
    plt.scatter(a_class['feat1'], a_class['feat2'], s=10)
    plt.scatter(b_class['feat1'], b_class['feat2'], color='orange', s=10)
    plt.plot(x1_value, x2_value, color='red')
    plt.xlabel('feat1')
    plt.ylabel('feat2')
    plt.legend(['Decision Boundary', 'Class 0', 'Class 1'])
    plt.title('Logistic Regression')
    plt.show()

# 학습률과 반복 횟수 설정
learning_rate = 0.01
num_iterations = 40000

# 로지스틱 회귀 학습 및 시각화
logistic_regression_and_plot(data, feat1, feat2, 0.5, num_iterations)
