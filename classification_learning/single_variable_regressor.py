# 단순 회귀 분석 모델 (독립 변수를 하나만 사용)

import pickle
import numpy as np

from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt

# 입력 데이터 파일
input_file = 'data_singlevar_regr.txt'

# 데이터 읽기
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# 데이터 분리하기 (학습용, 테스트용)
num_training = int(0.8 * len(X))
num_text = len(X) - num_training

# 학습 데이터
X_train, y_train = X[:num_training], y[:num_training]

# 테스트 데이터
X_test, y_test = X[num_training:], y[num_training:]

# 선형 회귀 분석 모델 오브젝트 생성하기
regressor = linear_model.LinearRegression()

# 학습 데이터로 회귀 분석 모델 학습시키기
regressor.fit(X_train, y_train)

# 결과 예측하기
y_test_pred = regressor.predict(X_test)

# 출력 값을 그래프로 그리기
plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_test_pred, color='black', linewidth=4)
plt.xticks(())
plt.yticks(())
plt.show()

# GT ( 생산비용 ) 와 비교하는 방식으로 회귀 분석 모델의 성능을 측정한다.

print("Linear regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

# 생성한 모델을 나중에 다시 사용할 수 있도록 파일에 저장함
output_model_file = 'model.pkl'

# 모델 저장하기
with open(output_model_file, 'wb') as f:
    pickle.dump(regressor, f)

# 모델 불러오기
with open(output_model_file, 'rb') as f:
    regressor_model = pickle.load(f)

# 테스트 데이터에 대해 예측하기
y_test_pred_new = regressor_model.predict(X_test)
print("\n New mean absolute error = ", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))



