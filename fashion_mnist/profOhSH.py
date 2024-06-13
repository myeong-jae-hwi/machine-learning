import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

# ====================== 데이터 train이랑 valid 나누는 부분 ======================
def split_data(x_train, y_train, x_test, y_test):
  train_unique_labels = np.unique(y_train)  # [0 1 2 3 4 5 6 7 8 9]
  test_unique_labels = np.unique(y_test)  # [0 1 2 3 4 5 6 7 8 9]
  train_label_indices = [np.where(y_train == i)[0][:1000]
                         for i in train_unique_labels]
  x_train_data = np.concatenate([x_train[idx] for idx in train_label_indices])
  y_train_data = np.concatenate([y_train[idx] for idx in train_label_indices])

  valid_label_indices = [np.where(y_train == i)[0][1000:2000]
                         for i in train_unique_labels]
  x_valid_data = np.concatenate([x_train[idx] for idx in valid_label_indices])
  y_valid_data = np.concatenate([y_train[idx] for idx in valid_label_indices])

  test_label_indices = [np.where(y_test == i)[0][0:200]
                        for i in test_unique_labels]
  x_test_data = np.concatenate([x_test[idx] for idx in test_label_indices])
  y_test_data = np.concatenate([y_test[idx] for idx in test_label_indices])

  return x_train_data, y_train_data, x_valid_data, y_valid_data, x_test_data, y_test_data
x_train_data, y_train_data, x_valid_data, y_valid_data, x_test_data, y_test_data = split_data(x_train, y_train, x_test, y_test)

# ====================== 데이터 시각화 부분 (매개변수에 넣은거 전체) ======================
def plot_images_per_class(data, num_images_per_class=10):
    total_images = len(data)
    num_full_windows = total_images // (10 * num_images_per_class)
    remaining_images = total_images % (10 * num_images_per_class)
    
    # 전체 데이터가 표시될 창의 개수
    num_windows = num_full_windows + (1 if remaining_images > 0 else 0)
    
    for window in range(num_windows):
        fig, axes = plt.subplots(10, num_images_per_class, figsize=(15, 15))

        # 만약 마지막 창이고 남은 이미지가 있는 경우
        if window == num_windows - 1 and remaining_images > 0:
            num_images = remaining_images
        else:
            num_images = 10 * num_images_per_class
        
        for i in range(num_images_per_class):
            class_data = data[window * 10 * num_images_per_class + i * num_images_per_class:
                              window * 10 * num_images_per_class + (i + 1) * num_images_per_class]
            for j in range(len(class_data)):
                axes[j, i].imshow(class_data[j])
                axes[j, i].axis('off')
        plt.show()


# ====================== 신발 클래스를 분류하고 싶었지만 부츠만 댐 ㅋㅋ =================================
def shoes_class(data, y_train):
    class_A = []
    class_A_labels = []
    class_shoes = []
    class_shoes_labels = []
    new_label = []

    for img, label in zip(data, y_train):
        # 데이터의 x축 기준 중간에 수직선을 그어 좌, 우의 0이 아닌 픽셀수를 검사
        mid_x = img.shape[1] // 2
        left_pixels = np.count_nonzero(img[:, :mid_x])
        right_pixels = np.count_nonzero(img[:, mid_x:])
        
        # 픽셀 수가 비슷하다면 class A에 저장, 그렇지 않다면 class_shoes에 저장
        if abs(left_pixels - right_pixels) < 80:
            class_A.append(img)
            class_A_labels.append(label)
            new_label.append(7)
        else:
            class_shoes.append(img)
            class_shoes_labels.append(label)

    class_A = np.array(class_A)
    class_A_labels = np.array(class_A_labels)
    class_shoes = np.array(class_shoes)
    class_shoes_labels = np.array(class_shoes_labels)

    return class_A, class_A_labels, class_shoes, class_shoes_labels
  
# =========================== 라벨 수 출력 ==========================

def matplot_label_counts(labels, title):
    label_counts = Counter(labels)
    labels, counts = zip(*label_counts.items())

    total_count = sum(counts)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, counts, color='skyblue')
    plt.xlabel('Labels')
    plt.ylabel('Counts')
    plt.title(title + ' EA : '+ str(total_count))

    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha='center', va='bottom')

    plt.show()
# =========================== 신발 분류 ==========================
def sneakers_class(data, y_train):
    sneakers = []
    class_B = []
    sneakers_labels = []
    class_B_label = []

    for img, label in zip(data, y_train):
        # 이미지의 높이와 너비를 가져옴
        height, width = img.shape[:2]
        top = 2
        bottom = 2

        # 이미지의 상단과 하단에서 10픽셀씩 잘라냄
        top_pixels = img[:top, :]
        bottom_pixels = img[height-bottom:, :]
        
        # 상단과 하단의 열이 모두 비어있으면 sneakers로 분류
        if np.sum(top_pixels) == 0 and np.sum(bottom_pixels) == 0:
            sneakers.append(img)
            sneakers_labels.append(label)
        else:
            class_B.append(img)
            class_B_label.append(label)

    sneakers=np.array(sneakers)
    sneakers_labels = np.array(sneakers_labels)
    class_B = np.array(class_B)
    class_B_label = np.array(class_B_label)

    return sneakers, sneakers_labels, class_B, class_B_label

# ===================== 바지 / 드레스 분류 =====================

def pants_dress_class(data, y_train):
    pants = []
    pants_labels = []
    class_C = []
    class_C_labels = []

    for img, label in zip(data, y_train):
        height, width = img.shape[:2]
        right = 5
        left = 5

        left_pixels = img[:, :left]
        right_pixels = img[:, width-right:]
        
        if np.sum(right_pixels) == 0 and np.sum(left_pixels) == 0:
            pants.append(img)
            pants_labels.append(label)
        else:
            class_C.append(img)
            class_C_labels.append(label)

    pants = np.array(pants)
    pants_labels = np.array(pants_labels)
    class_C = np.array(class_C)
    class_C_labels = np.array(class_C_labels)

    return pants, pants_labels, class_C, class_C_labels

# ===================== 티셔츠 분류 =======================

def t_shorts_class(data, y_train):
    t_short = []
    t_short_label = []
    class_D = []
    class_D_label = []

    for img, label in zip(data, y_train):
        height, width = img.shape[:2]

        right = 5
        left = 5

        bottom_nonzero_left = -1
        bottom_nonzero_right = -1

        for i in range(height-1, 1, -1):
            if img[i, left] >= 10:
                bottom_nonzero_left = i
                # img[bottom_nonzero_left, ] = 255
                break

        if bottom_nonzero_left != -1 and bottom_nonzero_left < 15:
            t_short.append(img)
            t_short_label.append(label)
        else:
            class_D.append(img)
            class_D_label.append(label)
        
    t_short = np.array(t_short)
    t_short_label = np.array(t_short_label)
    class_D = np.array(class_D)
    class_D_label = np.array(class_D_label) 

    return t_short, t_short_label, class_D, class_D_label

# 0이 연속으로 나오는 구역을 찾아냄
def zeros_area(arr):
    count = 0
    in_zeros = False
    for num in arr:
        if num == 0:
            if arr[num + 1] == 0:
                if not in_zeros:
                    in_zeros = True
                    count += 1
        else:
            in_zeros = False
    return count

# ===================== 바지 분류 =======================
def pants_func(data, y_train):
    pants_class = []
    pants_label = []
    class_E = []
    class_E_label = []

    for img, label in zip(data, y_train):
        height, width = img.shape[:2]
        bottom = 5

        feat_pants = height - bottom
        original_img = img.copy()

        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        fifth_col_zeros = zeros_area(img[feat_pants, ::-1]) 

        if fifth_col_zeros == 3:
            pants_class.append(img)
            pants_label.append(label)
        else:
            class_E.append(original_img)
            class_E_label.append(label)
        
    pants_class = np.array(pants_class)
    pants_label = np.array(pants_label)
    class_E = np.array(class_E)
    class_E_label = np.array(class_E_label)
    return pants_class, pants_label, class_E, class_E_label

# ===================== 가방 분류 =======================
def bag_class_fuc(data, y_train):
    bag_class = []
    bag_label = []
    class_F = []
    class_F_label = []

    for img, label in zip(data, y_train):
        height, width = img.shape[:2]
        top = 5
        feat_bag = top 

        fifth_col_zeros = zeros_area(img[top,:]) 

        if fifth_col_zeros >= 5:
            img[feat_bag, :] = 255
            bag_class.append(img)
            bag_label.append(label)
            
        else:
            class_F.append(img)
            class_F_label.append(label)
        
    bag_class = np.array(bag_class)
    bag_label = np.array(bag_label)
    class_F = np.array(class_F)
    class_F_label = np.array(class_F_label)

    return bag_class, bag_label, class_F, class_F_label

# ========================== 샌들이 안없어져서 재분류!!!!!!!!!!!!!!!!!!!!!! ============
def sandal_2(data,y_train):
    sandal = []
    sandal_label = []
    sneakers = []
    sneakers_label = []

    for img, label in zip(data, y_train):
        fifth_col_zeros = zeros_area(img[14,:]) 

        if fifth_col_zeros > 2:
            sandal.append(img)
            sandal_label.append(label)
        else:
            sneakers.append(img)
            sneakers_label.append(label)

    sneakers = np.array(sneakers)
    sneakers_label = np.array(sneakers_label)
    sandal = np.array(sandal)
    sandal_label = np.array(sandal_label)
    
    return sneakers, sneakers_label, sandal, sandal_label

# ========================= 샌들 분류 ======================
def sandal(data, y_train):
    sandal_class = []
    sandal_label = []
    class_G = []
    class_G_label = []
    pixel_sums=[]

    for img in data:
        binary_img  = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, 2)
        pixel_sum = np.sum(binary_img == 255)  
        pixel_sums.append(pixel_sum)

    threshold = np.percentile(pixel_sums, 10)
    
    for img, label, pixel_sum in zip(data, y_train, pixel_sums):
        if pixel_sum <= threshold:
            sandal_class.append(img)
            sandal_label.append(label)

        else:
            class_G.append(img)
            class_G_label.append(label)

    sandal_class = np.array(sandal_class)
    sandal_label = np.array(sandal_label)
    class_G = np.array(class_G)
    class_G_label = np.array(class_G_label)

    return sandal_class, sandal_label, class_G, class_G_label


# ====================== 클러치백 분류 =======================
def clutch_bag_fuc(data, y_train):
    
    def rectangle(image):
        binary_img  = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 111, 2)
        # 윤곽선 찾기
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # 윤곽선 근사화
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 근사화된 윤곽선이 사각형인지 확인
            if cv2.isContourConvex(approx):
                return True
        
        return False
    
    clutch_bag = []
    clutch_bag_label = []
    other_class = []
    other_label = []

    for img, label in zip(data, y_train):
        if rectangle(img):
            clutch_bag.append(img)
            clutch_bag_label.append(label)
        else:
            other_class.append(img)
            other_label.append(label)

    clutch_bag = np.array(clutch_bag)
    clutch_bag_label = np.array(clutch_bag_label)
    other_class = np.array(other_class)
    other_label = np.array(other_label)

    return clutch_bag, clutch_bag_label, other_class, other_label


def coat_class(data, y_train):
    coat = []
    coat_label = []
    other_class = []
    other_label = []
    
    pixel_sums = []

    for img, label in zip(data, y_train):
        binary_img  = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 91, 2)
        pixel_sum = np.sum(binary_img == 255)
        pixel_sums.append(pixel_sum)

    threshold = np.percentile(pixel_sums, 70)
    for img, label, pixel_sum in zip(data, y_train, pixel_sums):
        binary_img  = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 91, 2)

        if threshold >= pixel_sum:
            coat.append(binary_img)
            coat_label.append(label)

        else:
            other_class.append(binary_img)
            other_label.append(label)

    coat = np.array(coat)
    coat_label = np.array(coat_label)
    other_class = np.array(other_class)
    other_label = np.array(other_label)
    return coat, coat_label,other_class, other_label


# =================================================================
print(f"train data: {np.shape(x_train_data)}")
print(f"train data: {np.shape(x_valid_data)}")
print(f"test data: {np.shape(x_test_data)}")
sneakers, sneakers_label, class_A, class_A_label = sneakers_class(x_train_data, y_train_data)
class_B, class_B_label, class_shoes, class_shoes_label = shoes_class(class_A, class_A_label)
bag_class, bag_label, class_C, class_C_label = bag_class_fuc(class_B, class_B_label)

pants, pants_labels,class_D, class_D_label = pants_dress_class(class_C, class_C_label)

t_short, t_shorts_label, class_E, class_E_label = t_shorts_class(class_D, class_D_label)
pants_class, pants_label, dress_class, dress_label = pants_func(pants, pants_labels)
sandal_sneakers, sandal_sneakers_label, boots, boots_label = shoes_class(sneakers, sneakers_label)
boots = np.concatenate((class_shoes, boots))    # 부츠 합치기
boots_label = np.concatenate((class_shoes_label, boots_label))
sandal, sandal_label, sneakers, sneakers_label = sandal(sandal_sneakers, sandal_sneakers_label)
sneakers, sneakers_label, sandal_2, sandal_label_2 = sandal_2(sneakers, sneakers_label)
sandal = np.concatenate((sandal, sandal_2))
sandal_label = np.concatenate((sandal_label, sandal_label_2))
clutch_bag, clutch_bag_label, sneakers, sneakers_label = clutch_bag_fuc(sneakers, sneakers_label)
bag_class2, bag_label2, sneakers, sneakers_label = bag_class_fuc(sneakers, sneakers_label)

# print(np.shape(sneakers))
# plot_images_per_class(sneakers)

bag = np.concatenate((clutch_bag, bag_class, bag_class2))
bag_label = np.concatenate((clutch_bag_label, bag_label, bag_label2))
coat, coat_label, class_G, class_G_label = coat_class(class_E, class_E_label)

# matplot_label_counts(class_F_label,'class_G_label')

# matplot_label_counts(class_G_label,'coat_label')
# plot_images_per_class(sneakers)

# matplot_label_counts(t_shorts_label,'t_shorts')
# matplot_label_counts(pants_label,"pants_class")
# matplot_label_counts(class_E_label,"class_F")
# matplot_label_counts(dress_label,"dress_class")
# matplot_label_counts(sandal_label,"sandal_label")
# matplot_label_counts(sneakers_label,"sneakers")
# matplot_label_counts(bag_label, 'bag_class')
# matplot_label_counts(boots_label,"boots")

# plot_images_per_class(class_F)


# =================== 예측 레이블 적용 후 합치기 ========================
acc = 0

acc += np.count_nonzero(t_shorts_label == 0) #73%
acc += np.count_nonzero(pants_label == 1)   #91%
acc += np.count_nonzero(class_E_label == 2) #16%
acc += np.count_nonzero(dress_label == 3)   #81% 
acc += np.count_nonzero(sandal_label == 5)  #52%
acc += np.count_nonzero(sneakers_label == 7) #77%
acc += np.count_nonzero(bag_label == 8)     #71%
acc += np.count_nonzero(boots_label == 9)   #89%
print(f"정확도 : {acc/100}")

t_shorts_label = np.full_like(t_shorts_label, 0)
pants_label = np.full_like(pants_label, 1)
class_E_label = np.full_like(class_E_label, 2)
dress_label = np.full_like(dress_label, 3)
sandal_label = np.full_like(sandal_label, 5)
sneakers_label = np.full_like(sneakers_label, 7)
bag_label = np.full_like(bag_label, 8)
boots_label = np.full_like(boots_label, 9)

all_images = np.concatenate((t_short, pants_class, class_E, dress_class, sandal, sneakers, bag, boots))
all_labels = np.concatenate((t_shorts_label, pants_label, class_E_label, dress_label, sandal_label, sneakers_label, bag_label, boots_label))

# print("t_shorts_label: ",np.shape(t_short))
# print("pants_label: ",np.shape(pants))
# print("dress_label: ",np.shape(dress_class))
# print("sandal_label: ",np.shape(sandal))
# print("sneakers_label: ",np.shape(sneakers))
# print("bag_label: ",np.shape(bag))
# print("boots_label: ",np.shape(boots))
# print("class_F_label: ", np.shape(class_E))

# matplot_label_counts(all_labels, 'all_labels')


# 평탄화
all_images = all_images.reshape(all_images.shape[0],-1).astype('float32') / 255

x_train_data = x_train_data.reshape(-1, 28*28) / 255.0
x_valid_data = x_valid_data.reshape(-1, 28*28) / 255.0
x_test_data = x_test_data.reshape(-1, 28*28) / 255.0

C_values = [0.01]
max_depths = [10]
hidden_layer_sizes = [800]

accuracy_results = {
    "Logistic Regression": [],
    "Decision Tree": [],
    "MLP Classifier": []
}

# 로지스틱 
for C in C_values:
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(all_images, all_labels)
    predictions = model.predict(x_test_data)
    accuracy = accuracy_score(y_test_data, predictions)
    accuracy_results["Logistic Regression"].append((C, accuracy))
    print(accuracy_results)
    cm = confusion_matrix(predictions, y_test_data)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Logistic Regression (C={C}) Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# 결정 트리
# for depth in max_depths:
#     model = DecisionTreeClassifier(max_depth=depth)
#     model.fit(all_images, all_labels)
#     predictions = model.predict(x_test_data)
#     accuracy = accuracy_score(y_test_data, predictions)
#     accuracy_results["Decision Tree"].append((depth, accuracy))
#     print(accuracy_results)
#     cm = confusion_matrix(predictions, y_test_data)
#     plt.figure(figsize=(10, 7))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     plt.title(f'Decision Tree (max_depth={depth}) Confusion Matrix')
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.show()

# # # MLP 
# for hidden_layers in hidden_layer_sizes:
#     model = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=1000)
#     model.fit(all_images, all_labels)
#     predictions = model.predict(x_test_data)
#     accuracy = accuracy_score(y_test_data, predictions)
#     accuracy_results["MLP Classifier"].append((hidden_layers, accuracy))
#     print(accuracy_results)
#     cm = confusion_matrix(predictions, y_test_data)
#     plt.figure(figsize=(10, 7))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     plt.title(f'MLP Classifier (hidden_layers={hidden_layers}) Confusion Matrix')
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.show()

# 정확도 그래프 그리기
plt.figure(figsize=(15, 7))

# Logistic Regression 정확도 그래프
# C_vals, lr_accuracies = zip(*accuracy_results["Logistic Regression"])
# plt.plot(C_vals, lr_accuracies, marker='o', label='Logistic Regression')

# Decision Tree 정확도 그래프
# depth_vals, dt_accuracies = zip(*accuracy_results["Decision Tree"])
# plt.plot(depth_vals, dt_accuracies, marker='o', label='Decision Tree')

# # MLP Classifier 정확도 그래프
# hidden_layer_strs = [str(x) for x in hidden_layer_sizes]
# hidden_layer_vals, mlp_accuracies = zip(*accuracy_results["MLP Classifier"])
# plt.plot(hidden_layer_strs, mlp_accuracies, marker='o', label='MLP Classifier')

# plt.xlabel('Hyperparameters')
# plt.ylabel('Accuracy')
# plt.title('Model Accuracy by Hyperparameter')
# plt.legend()
# plt.show()
