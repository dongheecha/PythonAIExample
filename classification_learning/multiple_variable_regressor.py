# 다중 회귀 분석 모델 (독립 변수를 여러개 사용)

import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures
# 입력 데이터 파일
input_file = 'data_multivar_regr.txt'

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
linear_regressor = linear_model.LinearRegression()

# 학습 데이터로 회귀 분석 모델 학습시키기
linear_regressor.fit(X_train, y_train)

# 결과 예측하기
y_test_pred = linear_regressor.predict(X_test)

# GT ( 생산비용 ) 와 비교하는 방식으로 회귀 분석 모델의 성능을 측정한다.

print("Linear regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

# 차수가 10인 다항 회귀 모델을 생성하고 학습 데이터로 학습 시킨다. 그리고 샘플 데이터를 이용해 예측하는 부분을 작성한다.
polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)
datapoint = [[7.75, 6.35, 5.56]]
poly_datapoint = polynomial.fit_transform(datapoint)

poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)
print("\nLinear regression :\n", linear_regressor.predict(datapoint))
print("\nPolynomial regression:\n", poly_linear_model.predict(poly_datapoint))

