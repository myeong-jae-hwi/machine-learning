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
# import numpy as np
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings('ignore')

# def read_data(file_name):
#     data = np.loadtxt(file_name)
#     return data

# def sigmoid(num):
#     return 1 / (1 + np.exp(-num))

# def log_loss(y, prediction):
#     return -np.mean(y * np.log(prediction) + (1 - y) * np.log(1 - prediction))


# def train_logistic_regression(X, y, epochs, learning_rate,threshold_values, validX, validY):
#     # 매개변수 초기화
#     a, b, c, d = np.random.rand(4)
#     best_acc = 0
#     bestthreshold = None
#     acc_arr = []
#     acc_arr_test = []  # 훈련 손실 값 저장 리스트
#     acc_arr_valid = []  # 검증 손실 값 저장 리스트
    
#     for threshold in threshold_values:
#         for epoch in range(epochs):

#             # 모델 
#             z = a * X[:, 0] + b * X[:, 1] + c * X[:, 2] + d
#             prediction = sigmoid(z)

#             z_valid = a * validX[:, 0] + b * validX[:, 1] + c * validX[:, 2] + d
#             prediction_valid = sigmoid(z_valid)

#             loss_valid = log_loss(validY, prediction_valid)
            
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

#             predictions_valid = sigmoid(z_valid) >= 0.65

#             # acc 구하는 부분
#             predictions = prediction >= 0.65
#             acc = round(np.mean(predictions == y),3)
#             acc_arr.append((threshold, acc))

#             # print(f"test acc : {acc}")
#             # print(f"test Loss: {round((1 - acc),2)}")

#             accuracy_valid = round(np.mean(predictions_valid == validY),2)
#             # print(f"valid acc: {accuracy_valid}")
#             # print(f"valid Loss: {round((1 - accuracy_valid),2)}")
#             acc_arr_valid.append((1 - accuracy_valid))
#             acc_arr_test.append((1 - acc))
    
#         print(f"파라미터 업데이트 과정 :[{threshold}] ",round(a,2), round(b,2), round(c,2), round(d,2))
#         z_valid = a * validX[:, 0] + b * validX[:, 1] + c * validX[:, 2] + d

#         if acc > best_acc:
#             best_acc = acc
#             bestthreshold = threshold

#     TN = np.sum((y == 0) & (predictions == 0))
#     TP = np.sum((y == 1) & (predictions == 1))
#     FN = np.sum((y == 1) & (predictions == 0))
#     FP = np.sum((y == 0) & (predictions == 1))

#     # TP = 파란색이라고 예측했는데 파란색
#     # TN = 빨간색이라고 예측했는데 빨간색
#     # FP = 파란색이라고 예측했는데 빨간색 (경계 위에 있는 빨간색)
#     # FN = 빨간색이라고 예측했는데 파란색 (경계 아래에 있는 파란색)
#     print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
#     print(f"2 -3 : 혼동행렬을 그린 뒤 각각의 값을 나타내라. recall :{round(TP / (TP + FN),2)}, acc :{((TP + TN) / 600)}, precision :{round((TP/(TP+FP)),2)}")
#     print(f"Validation Loss: {loss_valid}")
#     print(len(acc_arr_test))
#     plt.figure(figsize=(12, 6))
#     plt.plot(acc_arr_valid, label='test loss', color='red')
#     plt.plot(acc_arr_test, label='valid loss',color='blue')
#     plt.show()

#     return bestthreshold, best_acc, acc_arr, round(a,2), round(b,2), round(c,2), round(d,2)

# # 파일에서 데이터 읽기
# testdata = read_data("test2.txt")
# validdata = read_data("valid2.txt")
# X = testdata[:, :3]
# y = testdata[:, 3]
# validX = validdata[:, :3]
# validY = validdata[:, 3]

# # 학습 파라미터 설정
# epochs = 500
# learning_rate = 0.01 
# threshold_values = np.linspace(0.1, 1, 10)

# bestthreshold, best_acc, acc_arr, a, b, c, d = train_logistic_regression(X, y, epochs, learning_rate,threshold_values, validX, validY)

# print(f"Best ξ: {bestthreshold}, Best Accuracy: {best_acc}")

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # 클래스에 따른 색상 지정
# color_map = {0: 'r', 1: 'b'}

# # 데이터셋에서 데이터 읽기 및 그리기
# x, y, z, label = testdata[:, 0], testdata[:, 1], testdata[:, 2], testdata[:, 3]
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

# # =----------------------3번 디씨전트리---------------------------========================
import numpy as np

# 데이터 로드
data = np.loadtxt('train2.txt')

# 특성과 타겟 변수 분리
X = data[:, :-1]
y = data[:, -1]

class DecisionTreeRegression:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return np.mean(y)

        num_features = X.shape[1]
        best_feature = None
        best_threshold = None
        best_loss = float('inf')

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold

                left_y = y[left_indices]
                right_y = y[right_indices]

                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                left_loss = np.mean((left_y - np.mean(left_y)) ** 2)
                right_loss = np.mean((right_y - np.mean(right_y)) ** 2)

                total_loss = left_loss + right_loss

                if total_loss < best_loss:
                    best_loss = total_loss
                    best_feature = feature
                    best_threshold = threshold

        if best_feature is None:
            return np.mean(y)

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return (best_feature, best_threshold, left_tree, right_tree)

    def predict(self, X):
        return np.array([self._predict(x, self.tree) for x in X])

    def _predict(self, x, tree):
        if isinstance(tree, np.float64):
            return tree

        feature, threshold, left_tree, right_tree = tree

        if x[feature] <= threshold:
            return self._predict(x, left_tree)
        else:
            return self._predict(x, right_tree)

# 모델 훈련
model = DecisionTreeRegression(max_depth=3)
model.fit(X, y)

# 훈련 데이터에서 예측값 계산
y_pred = model.predict(X)

# 정확도 계산
accuracy = np.mean((y - y_pred) ** 2)

# 결과 출력
print("훈련 데이터 정확도:", accuracy)

