# SVM 기법을 기반으로 주택 가격을 예측하는 서포트 벡터 회귀 분석 모델을 만드는 방법을 소개한다.

import numpy as np
from sklearn import datasets
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle

# 주택 데이터 가져오기
data = datasets.load_boston()

# 분석 결과가 치우치지 않도록 데이터를 섞는다
X, y = shuffle(data.data, data.target, random_state=7)

# 데이터를 학습용과 테스트용 80:20의 비율로 나눈다
num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# 선형 커널로 서포트 벡터 회귀 분석 모델을 생성한 후 학습시킨다. 여기서 C 매개변수는 학습 오류에 대한 패널티를
# 표현하다. C 값이 클수록 모델이 학습 데이터에 좀 더 최적화 되는데 C 값을 너무 크게 설정하면 오버피팅 현상이 발생해 제대로
# 일반화 할 수 있다. epsilon 매개변수는 임계점을 표현한다. 예측한 값이 GT 값의 범위 안에 있으면 오류에 대한 패널티를
# 부과하지 않는다.

# 서포트 벡터 회귀 모델 생성
sv_regressor = SVR(kernel='linear', C=1.0, epsilon=0.1)

# 서포트 벡터 회귀 모델 학습시키기
sv_regressor.fit(X_train, y_train)

# 서포트 벡터 회귀 분석 모델 성능 측정
y_text_pred = sv_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_text_pred)
evs = explained_variance_score(y_test, y_text_pred)
print("\b#### Performance ####")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))

# 테스트용 데이터포인트로 분석 모델 검증하기
test_data = [3.7, 0, 18.4, 1, 0.87, 5.95, 91, 2.5052, 26, 666, 20.2, 351.34, 15.27]
print("\nPredicted price:", sv_regressor.predict([test_data])[0])
