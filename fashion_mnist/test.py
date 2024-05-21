import numpy as np

def find_bottom_nonzero_indices(image):
    """
    이미지에서 각 열마다 값이 0이 아닌 가장 아래쪽 인덱스를 반환합니다.
    """
    height, width = image.shape[:2]
    bottom_indices = np.full(width, -1)  # 기본값 -1로 초기화 (값이 없는 경우)

    for x in range(width):
        column = image[:, x]
        nonzero_indices = np.where(column != 0)[0]
        if nonzero_indices.size > 0:
            bottom_indices[x] = nonzero_indices[-1]

    return bottom_indices

# 예시 사용
image = np.array([[0, 0, 0, 0],
                  [0, 255, 0, 0],
                  [0, 255, 0, 0],
                  [0, 0, 0, 0],
                  [0, 255, 0, 0]])

bottom_indices = find_bottom_nonzero_indices(image)
print(bottom_indices)
