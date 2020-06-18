#의사 결정 트리 기반 분류기 구축하기
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier

from ..Utils.utilities import visualize_classifier

# 입력 데이터 가져오기
input_file = 'data_decision_trees.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# 레이블을 기준으로 입력 데이터를 두 개의 클래스로 나눈다
class_0 = np.array(X[y==0])
class_1 = np.array(X[y==1])

# 스캐터 플롯을 이용해 입력 데이터를 시각화한다.
plt.fiqure()
plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='black', edgecolors='black', linewidths=1, marker='x')
plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white', edgecolors='black', linewidths=1, marker='o')
plt.title('Input data')

X_train, X_test, y_train, y_test = model_selection.train_text_split(X, y, test_size=0.25, random_state=5)

# 데이터를 학습 데이터셋과 테스트 데이터셋으로 나눈다.
# random_state 매개변수는 의사 결정 트리 기반 분류기 알고리즘 초기화를 위한 난수 생성기의 시드 값이다. max_depth 매개변수는 구축하려는 트리의 최대 깊이 값이다.

# 의사 결정 기반 분류기
params = {'random_state' : 0, 'max_depth': 4}
classifier = DecisionTreeClassifier(**params)
classifier.fit(X_train, y_train)
visualize_classifier(classifier, X_train, y_train, "Training dataset")

# 테스트 데이터에 대해 레이블을 예측하고 시각화함
y_test_pred = classifier.predict(X_test)
visualize_classifier(classifier, X_test, y_test, 'Test dataset')

# 분류리포트를 출력해 성능을 평가한다
class_names = ['Class-0', 'Class-1']
print("\n", + "#"*40)
print("\nClassifier performance on training dataset\n")
print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
print("#"*40 + "\n")
print("#"*40)
print("\nClassifier performance on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=class_names))
print("#"*40 + "\n")

plt.show()