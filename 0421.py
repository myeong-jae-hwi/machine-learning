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

# ========================== 2번 로지스틱 ========================================
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


# def train_logistic_regression(X, y, epochs, learning_rate,step_values, validX, validY):
#     # 매개변수 초기화
#     a, b, c, d = 0.1,0.2,0.3,0.4
#     best_acc = 0
#     beststep = None
#     acc_arr = []
#     acc_arr_test = []  # 훈련 손실 값 저장 리스트
#     acc_arr_valid = []  # 검증 손실 값 저장 리스트
#     abcd_arr = []
    
#     for step in step_values:
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
#             acc_arr.append((step, acc))

#             # print(f"test acc : {acc}")
#             # print(f"test Loss: {round((1 - acc),2)}")

#             accuracy_valid = round(np.mean(predictions_valid == validY),2)
#             abcd_arr.append((a, b, c, d))  # 현재 파라미터 저장
#             acc_arr_valid.append((1 - accuracy_valid))
#             acc_arr_test.append((1 - acc))
    
#         print(f"파라미터 업데이트 과정 :[{step}] ",round(a,2), round(b,2), round(c,2), round(d,2))
#         z_valid = a * validX[:, 0] + b * validX[:, 1] + c * validX[:, 2] + d

#         if acc > best_acc:
#             best_acc = acc
#             beststep = step

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

#     plt.figure(figsize=(12, 6))
#     plt.plot(acc_arr_valid, label='test loss', color='red')
#     plt.plot(acc_arr_test, label='valid loss',color='blue')

#     for i in range(1, len(acc_arr_valid)):
#         if acc_arr_valid[i] > acc_arr_valid[i-1]: 
#             plt.axvline(x=i, color='green', linestyle='--')
#             print(f"파라미터: a={round(abcd_arr[i][0],2)}, b={round(abcd_arr[i][1],2)}, c={round(abcd_arr[i][2],2)}, d={round(abcd_arr[i][3],2)}")

#     plt.show()

#     return beststep, best_acc, acc_arr, round(a,2), round(b,2), round(c,2), round(d,2)

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
# step_values = np.linspace(0.1, 1, 10)

# beststep, best_acc, acc_arr, a, b, c, d = train_logistic_regression(X, y, epochs, learning_rate,step_values, validX, validY)

# print(f"Best ξ: {beststep}, Best Accuracy: {best_acc}")

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

def gini_impurity(labels):
    unique_classes = np.unique(labels)
    total_samples = len(labels)

    gini = 1 

    # 각 클래스에 대해 지니 불순도 계산
    for cls in unique_classes:
        cls_count = np.sum(labels == cls)
        cls_prob = cls_count / total_samples
        gini -= cls_prob ** 2

    return gini

def Split_data(data, n):
    x_splits = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), n)
    y_splits = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), n)
    z_splits = np.linspace(np.min(data[:, 2]), np.max(data[:, 2]), n)

    max_impurity = -1
    max_impurity_axis = None
    max_impurity_region = None

    # 각 축에 대해 등분된 영역의 불순도 계산
    for axis, splits in zip(['x', 'y', 'z'], [x_splits, y_splits, z_splits]):
        for i in range(len(splits) - 1):
            if axis == 'x':
                indices = (data[:, 0] >= splits[i]) & (data[:, 0] < splits[i+1])
            elif axis == 'y':
                indices = (data[:, 1] >= splits[i]) & (data[:, 1] < splits[i+1])
            elif axis == 'z':
                indices = (data[:, 2] >= splits[i]) & (data[:, 2] < splits[i+1])

            sub_data = data[indices]
            if len(sub_data) == 0:
                impurity = 0  # 빈 영역은 불순도가 0
            else:
                # 불순도 계산
                impurity = gini_impurity(sub_data[:, 3])

            if impurity > max_impurity:
                max_impurity = impurity
                max_impurity_axis = axis
                max_impurity_region = i + 1

    return f"{max_impurity_axis},{max_impurity_region}"


data = np.loadtxt('test2.txt')
print(Split_data(data, 3))



import numpy as np
import matplotlib.pyplot as plt

def decision_tree(train_data, next_data=None, threshold=0.98, count=0):
    count += 1
    if next_data is None:
        next_data = train_data

    ori_feat1, ori_feat2, ori_feat3, ori_label = train_data[:, 0], train_data[:, 1], train_data[:, 2], train_data[:, 3] 
    feat1, feat2, feat3, label = next_data[:, 0], next_data[:, 1], next_data[:, 2], next_data[:, 3]
    n = 6
    feat1_min, feat1_max = np.min(feat1), np.max(feat1)
    feat2_min, feat2_max = np.min(feat2), np.max(feat2)
    feat3_min, feat3_max = np.min(feat3), np.max(feat3)
    division1 = np.linspace(feat1_min, feat1_max, n + 2)
    division2 = np.linspace(feat2_min, feat2_max, n + 2)
    division3 = np.linspace(feat3_min, feat3_max, n + 2)
    feat1_real = np.round(np.delete(division1, [0, len(division1) - 1]))
    feat2_real = np.round(np.delete(division2, [0, len(division2) - 1]))
    feat3_real = np.round(np.delete(division3, [0, len(division3) - 1]))
    impurity = []

    for i in range(n):

        #  f1 불순도
        f1_left_indices = feat1 <= feat1_real[i]
        f1_right_indices = feat1 > feat1_real[i]
        f1_left_labels = label[f1_left_indices]
        f1_right_labels = label[f1_right_indices]
        f1_left_class_counts = np.bincount(f1_left_labels.astype(int))
        f1_right_class_counts = np.bincount(f1_right_labels.astype(int))
        f1_left_impurity = 1 - np.max(f1_left_class_counts) / len(f1_left_labels)
        f1_right_impurity = 1 - np.max(f1_right_class_counts) / len(f1_right_labels)
        f1_impurity = (f1_left_impurity + f1_right_impurity) / 2

        #  f2 불순도
        f2_left_indices = feat2 <= feat2_real[i]
        f2_right_indices = feat2 > feat2_real[i]
        f2_left_labels = label[f2_left_indices]
        f2_right_labels = label[f2_right_indices]
        f2_left_class_counts = np.bincount(f2_left_labels.astype(int))
        f2_right_class_counts = np.bincount(f2_right_labels.astype(int))
        f2_left_impurity = 1 - np.max(f2_left_class_counts) / len(f2_left_labels)
        f2_right_impurity = 1 - np.max(f2_right_class_counts) / len(f2_right_labels)
        f2_impurity = (f2_left_impurity + f2_right_impurity) / 2

        #  f3 불순도
        f3_left_indices = feat3 <= feat3_real[i]
        f3_right_indices = feat3 > feat3_real[i]
        f3_left_labels = label[f3_left_indices]
        f3_right_labels = label[f3_right_indices]
        f3_left_class_counts = np.bincount(f3_left_labels.astype(int))
        f3_right_class_counts = np.bincount(f3_right_labels.astype(int))
        f3_left_impurity = 1 - np.max(f3_left_class_counts) / len(f3_left_labels)
        f3_right_impurity = 1 - np.max(f3_right_class_counts) / len(f3_right_labels)
        f3_impurity = (f3_left_impurity + f3_right_impurity) / 2

        impurity.append([f1_impurity, f2_impurity, f3_impurity])

    split_feature, split_threshold = np.unravel_index(np.argmin(impurity), (n, 3))
    if split_threshold == 0:
        predicted_labels = (feat1 > feat1_real[split_feature]).astype(int)
    elif split_threshold == 1:
        predicted_labels = (feat2 > feat2_real[split_feature]).astype(int)
    else:
        predicted_labels = (feat3 > feat3_real[split_feature]).astype(int)

    accuracy = np.sum(predicted_labels == label) / 600
    if accuracy < 0.5:
        accuracy = 1 - accuracy
    print(accuracy)

    if accuracy >= threshold:
        return count

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ori_feat1, ori_feat2, ori_feat3, cmap='bwr', c=ori_label, s=15)

    #======================================== 시각화 =====================================
    if split_threshold == 0:            # x축 기준
        yy, zz = np.meshgrid([feat2_min, feat2_max], [feat3_min, feat3_max])
        xx = np.full_like(yy, feat1_real[split_feature])
        ax.plot_surface(xx, yy, zz, color='green', alpha=0.3)
    elif split_threshold == 1:          # y축 기준
        xx, zz = np.meshgrid([feat1_min, feat1_max], [feat3_min, feat3_max])
        yy = np.full_like(xx, feat2_real[split_feature])
        ax.plot_surface(xx, yy, zz, color='black', alpha=0.3)
    else:                               # z축 기준 
        xx, yy = np.meshgrid([feat1_min, feat1_max], [feat2_min, feat2_max])
        zz = np.full_like(xx, feat3_real[split_feature])
        ax.plot_surface(xx, yy, zz, color='red', alpha=0.3)

    ax.scatter(feat1, feat2, feat3, cmap='bwr', c=label, s=15)
    plt.show()

    if split_threshold == 0:
        left_data = next_data[next_data[:, 0] < feat1_real[split_feature]]
        right_data = next_data[next_data[:, 0] >= feat1_real[split_feature]]
    elif split_threshold == 1:
        left_data = next_data[next_data[:, 1] < feat2_real[split_feature]]
        right_data = next_data[next_data[:, 1] >= feat2_real[split_feature]]
    else:
        left_data = next_data[next_data[:, 2] < feat3_real[split_feature]]
        right_data = next_data[next_data[:, 2] >= feat3_real[split_feature]]

    if len(left_data) > 0:
        decision_tree(train_data, next_data=left_data, threshold=0.99, count=count)
    if len(right_data) > 0:
        decision_tree(train_data, next_data=right_data, threshold=0.99, count=count)

train_data = np.loadtxt('valid2.txt')
count = decision_tree(train_data)
print(f"Depth: {count}")

