import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns

'''
코드 설명

1. 데이터 클래스 별 1000개씩 분할 -> train, valid, test (2000개)
2. 특징점에 따라 데이터를 분류
3. 분류된 데이터에 예측 레이블 적용
4. True 레이블과 비교해서 정확도 측정

적용한 특징점
T-shirt: 왼쪽에서 5픽셀을 기준으로 가장 큰 y축 즉, 가장 아래에 있는 픽셀이 15픽셀 이하일 경우 T-shirt로 분류
Pants: 아래에서 5픽셀을 기준으로 오른쪽에서 5번째 픽셀부터 3번 연속 0이 나오면 Pants로 분류


'''

x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

# def plot_images_per_label(data, labels, num_images_per_label=10):
#     unique_labels = np.unique(labels)
#     num_labels = len(unique_labels)
    
#     fig, axes = plt.subplots(num_images_per_label, num_labels, figsize=(num_labels * 2, num_images_per_label * 2))
    
#     for col, label in enumerate(unique_labels):
#         label_indices = np.where(labels == label)[0][:num_images_per_label]
#         label_data = data[label_indices]
        
#         for row in range(num_images_per_label):
#             axes[row, col].imshow(label_data[row])
#             axes[row, col].axis('off')
#             if row == 0:
#                 axes[row, col].set_title(f'Label {label}')
    
#     plt.tight_layout()
#     plt.show()

# plot_images_per_label(x_train, y_train)

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
def print_label_counts(labels, name):
    label_counts = Counter(labels)
    print(name)
    for label, count in label_counts.items():
        print(f"{label}: {count}개")

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

        # img[:, left] = 255 
        
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
def pants_class(data, y_train):
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
            # img[feat_pants, ] = 255
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
def bag_class(data, y_train):
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

def sandle(data, y_train):
    sandle_class = []
    sandle_label = []
    class_G = []
    class_G_label = []
    pixel_sums=[]

    for img in data:
        binary_img  = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 111, 2)
        pixel_sum = np.sum(binary_img == 255)  
        pixel_sums.append(pixel_sum)

    threshold = np.percentile(pixel_sums, 10)
    
    for img, label, pixel_sum in zip(data, y_train, pixel_sums):
        if pixel_sum <= threshold:
            sandle_class.append(img)
            sandle_label.append(label)
        else:
            class_G.append(img)
            class_G_label.append(label)

    sandle_class = np.array(sandle_class)
    sandle_label = np.array(sandle_label)
    class_G = np.array(class_G)
    class_G_label = np.array(class_G_label)

    return sandle_class, sandle_label, class_G, class_G_label


# ====================== 클러치백 분류 =======================
def clutch_bag(data, y_train):
    
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



# =================================================================

sneakers, sneakers_label, class_A, class_A_label = sneakers_class(x_train_data, y_train_data)
class_B, class_B_label, class_shoes, class_shoes_label = shoes_class(class_A, class_A_label)
pants, pants_labels,class_C, class_C_label = pants_dress_class(class_B, class_B_label)
t_short, t_shorts_label, class_D, class_D_label = t_shorts_class(class_C, class_C_label)
pants_class, pants_label, dress_class, dress_label = pants_class(pants, pants_labels)
sandle_sneakers, sandle_sneakers_label, boots, boots_label = shoes_class(sneakers, sneakers_label)
boots = np.concatenate((class_shoes, boots))    # 부츠 합치기
boots_label = np.concatenate((class_shoes_label, boots_label))
sandle, sandle_label, sneakers, sneakers_label = sandle(sandle_sneakers, sandle_sneakers_label)
clutch_bag, clutch_bag_label, sneakers, sneakers_label = clutch_bag(sneakers, sneakers_label)
bag_class, bag_label, class_F, class_F_label = bag_class(class_D, class_D_label)
bag = np.concatenate((clutch_bag, bag_class))
bag_label = np.concatenate((clutch_bag_label, bag_label))


matplot_label_counts(t_shorts_label,'t_shorts')
matplot_label_counts(pants_label,"pants_class")
matplot_label_counts(class_F_label,"class_F")
matplot_label_counts(dress_label,"dress_class")
matplot_label_counts(sandle_label,"sandle_label")
matplot_label_counts(sneakers_label,"sneakers")
matplot_label_counts(bag_label, 'bag_class')
matplot_label_counts(boots_label,"boots")

plot_images_per_class(sneakers)


# =================== 예측 레이블 적용 후 합치기 ========================
t_shorts_label = np.full_like(t_shorts_label, 0)
pants_label = np.full_like(pants_label, 1)
# class_F_label = np.full_like(class_F_label, 2)
dress_label = np.full_like(dress_label, 3)
sandle_label = np.full_like(sandle_label, 5)
sneakers_label = np.full_like(sneakers_label, 7)
bag_label = np.full_like(bag_label, 8)
boots_label = np.full_like(boots_label, 9)

all_images = np.concatenate((t_short, pants_class, class_F, dress_class, sandle, sneakers, bag, boots))
all_labels = np.concatenate((t_shorts_label, pants_label, class_F_label, dress_label, sandle_label, sneakers_label, bag_label, boots_label))

print("t_shorts_label: ",np.shape(t_short))
print("pants_label: ",np.shape(pants))
print("dress_label: ",np.shape(dress_class))
print("sandle_label: ",np.shape(sandle))
print("sneakers_label: ",np.shape(sneakers))
print("bag_label: ",np.shape(bag))
print("boots_label: ",np.shape(boots))
print("class_F_label: ", np.shape(class_F))

# print("class_shoes_label: ",np.shape(class_shoes))

matplot_label_counts(all_labels, 'all_labels')

print(accuracy_score(y_train_data, all_labels))




# 모델 입력 크기에 맞게 데이터 형태 변환
print(np.shape(all_images))
print("label: ",np.shape(all_labels))
all_images = all_images.reshape(all_images.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

# 모델 입력 크기에 맞게 데이터 형태 변환
all_images = all_images.reshape(all_images.shape[0], -1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], -1).astype('float32') / 255

# 로지스틱
logistic_model = LogisticRegression()
logistic_model.fit(all_images, all_labels)
logistic_pred = logistic_model.predict(x_test)
logistic_accuracy = accuracy_score(y_test, logistic_pred)
print("Logistic Regression Accuracy:", logistic_accuracy)

# 결정 트리 
tree_model = DecisionTreeClassifier()
tree_model.fit(all_images, all_labels)
tree_pred = tree_model.predict(x_test)
tree_accuracy = accuracy_score(y_test, tree_pred)
print("Decision Tree Accuracy:", tree_accuracy)

# MLP
mlp_model = MLPClassifier()
mlp_model.fit(all_images, all_labels)
mlp_pred = mlp_model.predict(x_test)
mlp_accuracy = accuracy_score(y_test, mlp_pred)
print("MLP Accuracy:", mlp_accuracy)


# =========== 시각화 하는 부분 ============
# print("\nClass pants : ")
# matplot_label_counts(pants_label, "pants")
# print("\nClass dress : ")
# matplot_label_counts(dress_label, "dress")
# print("\nClass t_short : ")
# matplot_label_counts(t_shorts_label, "t_shorts")
# print("\nClass class_D : ")
# matplot_label_counts(class_D_label, "class_D")
# print("\nClass sandle_sneakers : ")
# matplot_label_counts(sandle_sneakers_label,"sandle_sneakers")
# print("\nClass boots : ")
# matplot_label_counts(boots_label, "boots")

# print(np.shape(class_D))
# print_label_counts(class_D_label, 'class_D_label')
# print(np.shape(bag_class))
# print_label_counts(bag_label, 'bag_label')
# print_label_counts(boots_label, "boots")
# plot_images_per_class(boots)



# print(np.shape(class_shoes))
# print(np.shape(pants))
