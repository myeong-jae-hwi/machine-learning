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
        for i in range(0, 600, 50):
            # 훈련 데이터에서 데이터 포인트 선택
            X_point = X_train[i]
            y_point = y_train[i]
            
            # 해당 데이터 포인트에 대한 예측
            z_train = a * X_point[0] + b * X_point[1] + c * X_point[2] + d
            prediction_train = sigmoid(z_train)

            z_valid = a * X_valid[i, 0] + b * X_valid[i, 1] + c * X_valid[i, 2] + d
            prediction_valid = sigmoid(z_valid)
            
            
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


# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D 

# # Gini 불순도 계산
# def gini_impurity(labels):
#     unique_labels, counts = np.unique(labels, return_counts=True)
#     impurity = 1.0
#     for count in counts:
#         prob_of_lbl = count / float(len(labels))
#         impurity -= prob_of_lbl**2
#     return impurity

# # 데이터 분할
# def split_data(data, feature_index, threshold):
#     left = data[data[:, feature_index] <= threshold]
#     right = data[data[:, feature_index] > threshold]
#     return left, right

# # 최적 분할 찾기
# def find_best_split(data, num_features):
#     best_impurity = 1
#     best_criteria = None
#     best_splits = None
    
#     for feature_index in range(num_features):
#         thresholds = np.unique(data[:, feature_index])
#         for threshold in thresholds:
#             left, right = split_data(data, feature_index, threshold)
#             if len(left) > 0 and len(right) > 0:
#                 # 가중치가 적용된 Gini 불순도 계산
#                 w_impurity = (len(left) * gini_impurity(left[:, -1]) + len(right) * gini_impurity(right[:, -1])) / len(data)
#                 if w_impurity < best_impurity:
#                     best_impurity = w_impurity
#                     best_criteria = (feature_index, threshold)
#                     best_splits = (left, right)
                    
#     return best_criteria, best_splits

# # 결정 경계 그리기
# def plot_decision_boundary(data, best_criteria, z_data):
#     color_map = {0: 'r', 1: 'b'}

#     x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
#     y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
#                          np.arange(y_min, y_max, 0.1))
    
#     Z = np.array([best_criteria[0] if x[best_criteria[0]] <= best_criteria[1] else 1-best_criteria[0] for x in np.c_[xx.ravel(), yy.ravel()]])
#     Z = Z.reshape(xx.shape)
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.contourf(xx, yy, Z, alpha=0.4)

#     # 데이터 점 그리기
#     label = data[:, 2]
#     colors = [color_map[int(i)] for i in label]

#     ax.scatter(data[:, 0], data[:, 1], z_data, c=colors, s=20)

#     plt.show()





# # 파일에서 데이터 읽어오기
# def read_data(file_name):
#     data = np.loadtxt(file_name, delimiter=' ')
#     return data

# # 데이터 준비 및 결정 경계 그리기
# data = read_data("valid2.txt") 
# z_data = data[:, 2]
# data = np.delete(data, 2, 1) 

# # 최적 분할 찾기
# best_criteria, best_splits = find_best_split(data, 2)
# print("Best Criteria:", best_criteria)

# # 결정 경계 그리기
# plot_decision_boundary(data, best_criteria, z_data)

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Gini 불순도 계산
# def gini_impurity(labels):
#     _, counts = np.unique(labels, return_counts=True)
#     impurity = 1.0
#     for count in counts:
#         prob_of_lbl = count / len(labels)
#         impurity -= prob_of_lbl ** 2
#     return impurity

# # 데이터 분할
# def split_data(data, feature_index, threshold):
#     left = data[data[:, feature_index] <= threshold]
#     right = data[data[:, feature_index] > threshold]
#     return left, right

# # 최적 분할 찾기
# def find_best_split(data):
#     best_impurity = 1
#     best_criteria = None
    
#     # 각 특징에 대해 최적 분할 찾기
#     for feature_index in range(data.shape[1] - 1):
#         thresholds = np.unique(data[:, feature_index])
#         for threshold in thresholds:
#             left, right = split_data(data, feature_index, threshold)
#             if len(left) > 0 and len(right) > 0:
#                 impurity = (len(left) * gini_impurity(left[:, -1]) + len(right) * gini_impurity(right[:, -1])) / len(data)
#                 if impurity < best_impurity:
#                     best_impurity = impurity
#                     best_criteria = {'feature_index': feature_index, 'threshold': threshold}
    
#     return best_criteria

# # 결정 경계 평면화
# def flatten_decision_boundary(data, decision_boundary):
#     return np.array([[point[0], point[1], decision_boundary['feature_index']] for point in data])

# # 3차원 데이터 시각화
# def plot_3d_scatter(data, decision_boundary):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=data[:, 2], cmap='coolwarm', s=20)

#     # 결정 경계 평면 그리기
#     xx, yy = np.meshgrid(np.linspace(data[:,0].min(), data[:,0].max(), 100),
#                          np.linspace(data[:,1].min(), data[:,1].max(), 100))
#     zz = np.ones(xx.shape) * decision_boundary['threshold']
#     ax.plot_surface(xx, yy, zz, alpha=0.5, color='grey')

#     plt.show()

# # 데이터 읽기
# data = np.loadtxt("valid2.txt", delimiter=' ')
# x_data = data[:, :2]  # z 축을 제외한 데이터
# z_data = data[:, 2]   # z 축 데이터

# # 최적 분할 찾기
# best_criteria = find_best_split(data)
# print("Best Criteria:", best_criteria)

# # 결정 경계 평면화
# decision_boundary = {'feature_index': best_criteria['feature_index'], 'threshold': best_criteria['threshold']}
# flattened_decision_boundary = flatten_decision_boundary(x_data, decision_boundary)

# # 시각화
# plot_3d_scatter(data, decision_boundary)

# def filter_data(data, decision_boundary):
#     filtered_data = []
#     for point in data:
#         if point[decision_boundary['feature_index']] <= decision_boundary['threshold']:
#             filtered_data.append(point)
#     return np.array(filtered_data)

# # 데이터 필터링
# filtered_data = filter_data(data, decision_boundary)

# # 필터링된 데이터 산점도 시각화
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# colors = ['r' if point[2] == 0 else 'b' for point in filtered_data]
# ax.scatter(filtered_data[:, 0], filtered_data[:, 1], filtered_data[:, 2], c=colors, s=20)

# plt.show()
