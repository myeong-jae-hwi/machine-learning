import numpy as np
import matplotlib.pyplot as plt

x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

# def plot_images_per_label(data, labels, num_images_per_label=60):
#     unique_labels = np.unique(labels)
#     num_labels = len(unique_labels)

#     for label in unique_labels:
#         label_indices = np.where(labels == label)[0][:num_images_per_label]
#         label_data = data[label_indices]
#         num_rows = (num_images_per_label + 9) // 10  # Calculate number of rows needed
#         fig, axes = plt.subplots(num_rows, 10, figsize=(15, num_rows * 1.5))
#         for i in range(num_images_per_label):
#             row = i // 10
#             col = i % 10
#             axes[row, col].imshow(label_data[i])
#             axes[row, col].axis('off')
#         plt.show()

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


def shoes_class(data):
    class_A = []
    class_shoes = []

    for img in data:
        # Step 3: 데이터의 x축 기준 중간에 수직선을 그어 좌, 우의 0이 아닌 픽셀수를 검사
        mid_x = img.shape[1] // 2
        left_pixels = np.count_nonzero(img[:, :mid_x])
        right_pixels = np.count_nonzero(img[:, mid_x:])
        
        # Step 4: 픽셀 수가 비슷하다면 class A에 저장, 그렇지 않다면 class_shoes에 저장
        if abs(left_pixels - right_pixels) < 80:
            class_A.append(img)
        else:
            class_shoes.append(img)

    return class_A, class_shoes
  
class_A, class_shoes = shoes_class(x_train_data)
print(np.shape(class_shoes))
plot_images_per_class(class_shoes)

