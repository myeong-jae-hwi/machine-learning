# 
# import numpy as np
# import matplotlib.pyplot as plt

# cnt = 0

# def data_split(dataset):
#     return dataset[:, :-1], dataset[:, -1].astype(int)

# def calculate_accuracy(y_true, y_pred):
#     return np.bincount(y_true == y_pred)[1] / len(y_true)

# def log_loss(y_true, y_pred):
#     y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
#     loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
#     return loss

# def _stopping_criteria(y, depth, max_depth, min_samples_split):
#     if len(np.unique(y)) == 1:
#         return True
#     if depth == max_depth:
#         return True
#     if len(y) < min_samples_split:
#         return True
#     return False

# def _build_tree(X, y, depth, max_depth, min_samples_split, sample_weights):
#     global cnt
#     if _stopping_criteria(y, depth, max_depth, min_samples_split):
#         leaf_value = _calculate_leaf_value(y, sample_weights)
#         print(f"여기로 왔어요:{leaf_value}")
#         return leaf_value

#     feature_index, threshold = _best_split(X, y, sample_weights, min_samples_split)
#     if feature_index is None:
#         leaf_value = _calculate_leaf_value(y, sample_weights)
#         print(f"여기로 왔어요2:{leaf_value}")
#         return leaf_value

#     left_indices = X[:, feature_index] <= threshold
#     right_indices = X[:, feature_index] > threshold

#     X_left, X_right = X[left_indices], X[right_indices]
#     y_left, y_right = y[left_indices], y[right_indices]

#     weights_left = sample_weights[left_indices]
#     weights_right = sample_weights[right_indices]

#     left_subtree = _build_tree(X_left, y_left, depth + 1, max_depth, min_samples_split, weights_left)
#     right_subtree = _build_tree(X_right, y_right, depth + 1, max_depth, min_samples_split, weights_right)
    
#     cnt += 1

#     print(f"cnt: {cnt}\n{left_subtree}")
#     print(f"{threshold}\n")

#     return (feature_index, threshold, left_subtree, right_subtree)

# def _calculate_leaf_value(y, sample_weights) -> dict:
#     classes, counts = np.unique(y, return_counts=True)
#     weighted_counts = np.zeros_like(counts, dtype=np.float64)

#     for i, cl in enumerate(classes):
#         weighted_counts[i] = sample_weights[y == cl].sum()

#     return dict(zip(classes, weighted_counts))

# def _best_split(X, y, sample_weights, min_samples_split):
#     best_gain = 0
#     best_feature, best_threshold = None, None
#     current_impurity = _calculate_impurity(y, sample_weights)

#     for feature_index in range(X.shape[1]):
#         thresholds = np.unique(X[:, feature_index])
#         for threshold in thresholds:
#             left_indices = X[:, feature_index] <= threshold
#             right_indices = X[:, feature_index] > threshold

#             if (
#                 np.sum(left_indices) < min_samples_split
#                 or np.sum(right_indices) < min_samples_split
#             ):
#                 continue

#             gain = _calculate_gain(
#                 y,
#                 sample_weights,
#                 left_indices,
#                 right_indices,
#                 current_impurity,
#             )
#             if gain > best_gain:
#                 best_gain = gain
#                 best_feature = feature_index
#                 best_threshold = threshold

#     return best_feature, best_threshold

# def _calculate_impurity(y, sample_weights) -> float:
#     total_weight = np.sum(sample_weights)
#     _, counts = np.unique(y, return_counts=True)
#     weighted_counts = np.zeros_like(counts, dtype=np.float64)

#     for i, cl in enumerate(np.unique(y)):
#         weighted_counts[i] = np.sum(sample_weights[y == cl])

#     prob = weighted_counts / total_weight
#     gini = 1 - np.sum(prob**2)

#     return gini

# def _calculate_gain(y, sample_weights, left_indices, right_indices, current_impurity):
#     left_weight = np.sum(sample_weights[left_indices])
#     right_weight = np.sum(sample_weights[right_indices])
#     total_weight = left_weight + right_weight

#     weights_left = sample_weights[left_indices]
#     weights_right = sample_weights[right_indices]

#     left_impurity = _calculate_impurity(y[left_indices], weights_left)
#     right_impurity = _calculate_impurity(y[right_indices], weights_right)

#     weighted_impurity = (left_weight / total_weight) * left_impurity
#     weighted_impurity += (right_weight / total_weight) * right_impurity
#     gain = current_impurity - weighted_impurity

#     return gain

# def _predict_single(node, x):
#     while isinstance(node, tuple):
#         feature_index, threshold, left_subtree, right_subtree = node
#         if x[feature_index] <= threshold:
#             node = left_subtree
#         else:
#             node = right_subtree

#     return max(node, key=node.get)

# def fit(X, y, max_depth=10, criterion="gini", min_samples_split=2, min_samples_leaf=1, max_features=None, min_impurity_decrease=0.0, max_leaf_nodes=None, random_state=None, sample_weights=None):
#     if sample_weights is None:
#         sample_weights = np.ones(len(y))
#     sample_weights = np.array(sample_weights)
#     root = _build_tree(X, y, 0, max_depth, min_samples_split, sample_weights)
#     # print(root)
#     return root

# def predict(root, X):
#     preds = [_predict_single(root, x) for x in X]
#     return np.array(preds)

# train_set = np.loadtxt("train2.txt")
# valid_set = np.loadtxt("valid2.txt")
# test_set = np.loadtxt("test2.txt")

# X_train, y_train = data_split(dataset=train_set)
# X_test, y_test = data_split(dataset=test_set)
# X_valid, y_valid = data_split(dataset=valid_set)

# depth_range = range(2, 11)
# train_loss = []
# valid_loss = []
# for depth in depth_range:
#     tree = fit(X_train, y_train, max_depth=depth)
#     train_acc = log_loss(
#         y_true=y_train,
#         y_pred=predict(tree, X_train),
#     )
#     valid_acc = log_loss(
#         y_true=y_valid,
#         y_pred=predict(tree, X_valid),
#     )

#     train_loss.append(train_acc)
#     valid_loss.append(valid_acc)

# plt.plot(list(depth_range), train_loss, label="train acc")
# plt.plot(list(depth_range), valid_loss, label="valid acc")
# plt.xlabel("Max Depths")
# plt.ylabel("Log-Loss")
# plt.title("Loss Over Various Depths")
# plt.legend()
# plt.grid(alpha=0.2)
# plt.tight_layout()
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt


# def decision_tree(train_data, next_data=None ,threshold=0.98, count=0):
#     count += 1
#     feat1, feat2, feat3, label = train_data[:,0], train_data[:, 1], train_data[:, 2], train_data[:, 3]
#     data_n = len(train_data)
#     n = 6
#     feat1_min, feat1_max = np.min(feat1), np.max(feat1)
#     feat2_min, feat2_max = np.min(feat2), np.max(feat2)
#     feat3_min, feat3_max = np.min(feat3), np.max(feat3)
#     division1 = np.linspace(feat1_min, feat1_max, n+2)
#     division2 = np.linspace(feat2_min, feat2_max, n+2)
#     division3 = np.linspace(feat3_min, feat3_max, n+2)
#     feat1_real = np.round(np.delete(division1, [0, len(division1)-1]))
#     feat2_div = np.round(np.delete(division2, [0, len(division2)-1]))
#     feat3_div = np.round(np.delete(division3, [0, len(division3)-1]))
#     impurity = []

#     # 불순도 계산 
#     for i in range(n):
#         f1_left_indices = feat1 <= feat1_div[i]
#         f1_right_indices = feat1 > feat1_div[i]
#         f1_left_labels = label[f1_left_indices]
#         f1_right_labels = label[f1_right_indices]
#         f1_left_class_counts = np.bincount(f1_left_labels.astype(int))
#         f1_right_class_counts = np.bincount(f1_right_labels.astype(int))
#         f1_left_impurity = 1 - np.max(f1_left_class_counts) / len(f1_left_labels)
#         f1_right_impurity = 1 - np.max(f1_right_class_counts) / len(f1_right_labels)
#         f1_impurity = (f1_left_impurity + f1_right_impurity) / 2

#         f2_left_indices = feat2 <= feat2_div[i]
#         f2_right_indices = feat2 > feat2_div[i]
#         f2_left_labels = label[f2_left_indices]
#         f2_right_labels = label[f2_right_indices]
#         f2_left_class_counts = np.bincount(f2_left_labels.astype(int))
#         f2_right_class_counts = np.bincount(f2_right_labels.astype(int))
#         f2_left_impurity = 1 - np.max(f2_left_class_counts) / len(f2_left_labels)
#         f2_right_impurity = 1 - np.max(f2_right_class_counts) / len(f2_right_labels)
#         f2_impurity = (f2_left_impurity + f2_right_impurity) / 2

#         f3_left_indices = feat3 <= feat3_div[i]
#         f3_right_indices = feat3 > feat3_div[i]
#         f3_left_labels = label[f3_left_indices]
#         f3_right_labels = label[f3_right_indices]
#         f3_left_class_counts = np.bincount(f3_left_labels.astype(int))
#         f3_right_class_counts = np.bincount(f3_right_labels.astype(int))
#         f3_left_impurity = 1 - np.max(f3_left_class_counts) / len(f3_left_labels)
#         f3_right_impurity = 1 - np.max(f3_right_class_counts) / len(f3_right_labels)
#         f3_impurity = (f3_left_impurity + f3_right_impurity) / 2

#         impurity.append([f1_impurity, f2_impurity, f3_impurity])
#         # print(impurity)

#     split_feature, split_threshold = np.unravel_index(np.argmin(impurity), (n, 3))
#     if split_threshold == 0:
#         predicted_labels = (feat1 > feat1_div[split_feature]).astype(int)
#     elif split_threshold == 1:
#         predicted_labels = (feat2 > feat2_div[split_feature]).astype(int)
#     else:
#         predicted_labels = (feat3 > feat3_div[split_feature]).astype(int)

#     ##################### acc 코드 수정 ######################
#     accuracy = np.sum(predicted_labels == label) / 600
#     # accuracy = np.mean(predicted_labels == label)
#     if (accuracy < 0.5):
#         accuracy = 1-accuracy
#     print(accuracy)
#     ########################################################
#     if accuracy >= threshold:
#         return count

#     if split_threshold == 0:
#         left_data = train_data[train_data[:, 0] <= feat1_div[split_feature]]
#         right_data = train_data[train_data[:, 0] > feat1_div[split_feature]]
#     elif split_threshold == 1:
#         left_data = train_data[train_data[:, 1] <= feat2_div[split_feature]]
#         right_data = train_data[train_data[:, 1] > feat2_div[split_feature]]
#     else:
#         left_data = train_data[train_data[:, 2] <= feat3_div[split_feature]]
#         right_data = train_data[train_data[:, 2] > feat3_div[split_feature]]

#     left_impurity = calculate_impurity(left_data)
#     right_impurity = calculate_impurity(right_data)

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(feat1, feat2, feat3, cmap='bwr', c=label, s=15)

#     ############## 시각화 ################
#     if split_threshold == 0:
#         yy, zz = np.meshgrid([feat2_min, feat2_max], [feat3_min, feat3_max])
#         xx = np.full_like(yy, feat1_div[split_feature])
#         ax.plot_surface(xx, yy, zz, color='green', alpha=0.3)
#     elif split_threshold == 1:
#         xx, zz = np.meshgrid([feat1_min, feat1_max], [feat3_min, feat3_max])
#         yy = np.full_like(xx, feat2_div[split_feature])
#         ax.plot_surface(xx, yy, zz, color='green', alpha=0.3)
#     else:
#         xx, yy = np.meshgrid([feat1_min, feat1_max], [feat2_min, feat2_max])
#         zz = np.full_like(xx, feat3_div[split_feature])
#         ax.plot_surface(xx, yy, zz, color='green', alpha=0.3)

#     ax.scatter(feat1, feat2, feat3, cmap='bwr', c=label, s=15)
#     plt.show()

#     if left_impurity >= right_impurity:
#         return decision_tree(train_data, left_data, threshold=0.99, count=count)
#     else:
#         return decision_tree(train_data, right_data, threshold=0.99, count=count)


# def calculate_impurity(data):
#     labels = data[:, -1]
#     label_count = np.bincount(labels.astype(int))
#     class_probabilities = label_count / len(labels)
#     impurity = 1 - np.max(class_probabilities)
#     return impurity


# train_data = np.loadtxt('train2.txt')
# count = decision_tree(train_data)
# print(f"Depth: {count}")


