# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.naive_bayes import GaussianNB
# from sklearn import cross_decomposition
#
# from ..Utils.utilities import  visualize_classifier
#
# # 데이터를 담은 입력 파일
# input_file = 'data_multivar_nb.txt'
#
# # 입력 파일에서 데이터 가져오기
# data = np.loadtxt(input_file, delimiter=',')
# X, y = data[:, :-1], data[:, -1]
#
# # 나이브 베이즈 분류기 생성
# classifier = GaussianNB # 가우시안 분포는 연속 확률 분포 중 하나이다.
#
# # 분류기 학습시키기
# classifier.fit(X, y)
#
# # 학습한 분류기로 예측한 결과 구하기
# y_red = classifier.predict(X)
#
# # 정확도 계산
# accuracy = 100.0 * (y == y_red).sum() / X.shape[0]
# print("Accuracyof Naive Bayes classifier =", round(accuracy, 2), "%")
#
# # 분류기 성능 시각화
# visualize_classifier(classifier, X, y)
#
# # 데이터를 학습용과 테스트용으로 나누기 (학습할 때 사용한 데이터로 검증하지 않도록 교차 검증 기법을 적용해야 한다.)
# #