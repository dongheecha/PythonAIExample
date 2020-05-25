import numpy as np
import matplotlib.pyplot as plt

def visualize_classifier(classifier, X, y) :
    # 메시 그리드에서 사용할 X와 y에 대한 최댓값과 최솟값을 정의 한다.
    # 메시 그리드는 함수를 평가하는 데 사용할 값들의 집합으로서 각각의 클래스의 경계 값을 시각적으로 표현하는 데 활용한다.
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    # 메시그리드를 그릴 때 적용할 간격 정의
    mesh_step_size = 0.01

    # 메시 그리드의 X 값과 Y 값 정의
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size),
                                 np.arange(min_y, max_y, mesh_step_size))

    # 메시 그리드에 대해 분류기 실행
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])

    # 결과로 나온 배열 정리
    output = output.reshape(x_vals.shape)

    # 그래프 생성
    plt.figure()

    # 그래프의 색상 지정
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)

    # 그래프에 학습용 점 그리기
    plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black', linewidths=1, cmap=plt.cm.Paired)

    # 그래프의 경계 지정
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())

    # x 축과 y 축에 대한 눈금을 표시한다
    plt.xticks((np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1),
                          1.0)))
    plt.yticks((np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1),
                          1.0)))

    plt.show()
