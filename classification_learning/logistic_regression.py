import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

from PythonAIExample.Utils.utilities import visualize_classifier

# 샘플 입력 데이터 정의
X = np.array([[3.1, 7.2], [4, 6.7], [2.9, 8], [5.1, 4.5], [6, 5],[5.6, 5], [3.3, 0.4], [3.9, 0.9], [2.8, 1], [0.5, 3.4],
              [1, 4], [0.6, 4.9]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

# 로지스틱 회귀 분류기 생성
# C 값을 높이면 분류 오류에 대한 패널티가 최적화되어 작용되지만 너무 크게 지정하면 학습 데이터에 치우치는 오버피팅이 발생해 제대로 분류
# 할 수 없다.
classfier = linear_model.LogisticRegression(solver='liblinear', C=1)

# 분류기 학습
classfier.fit(X, y)

# 분류기 성능 시각화
visualize_classifier(classfier, X, y)


