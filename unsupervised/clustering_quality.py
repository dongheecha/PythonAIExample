import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans

X = np.loadtxt('data_quality.txt', delimiter = ',')

# 입력 데이터 그래프 그리기
plt.figure()
plt.scatter(X[:, 0], X[:, 1], color='black', s=80, marker='o', facecolors='none')
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
plt.title('Input data')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# 변수 초기화
# values 배열운 최적의 군집 수를 찾을 때 까지 반복할 횟수를 담는데 사용된다
scores = []
values = np.arange(2, 10)

# values 에 있는 모든 값에 대해 루프를 돌면서 K-평균모델을 만든다

for num_clusters in values:
    # KMeans 군집화 모델 학습 시키기
    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
    kmeans.fit(X)
# 현재 군집화 모델에 대한 실루엣 지수를 구하기. 기준은 유클리드 거리고 지정함
score = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean', sample_size=len(X))

#현재 값에 대한 실루엣 지수를 화면에 출력
print("\nNumber of clusters =", num_clusters)
print("Silhouette score =", score)
scores.append(score)

# 전체 값에 대한 실루엣 지수를 그래프로 표현함.
plt.figure()
plt.bar(values, scores, width=0.7, color='black', align='center')
plt.title("Silhouette score VS number of clusters")

# 가장 높은 실루엣 지수를 찾아서 최적의 군집 수 구하기
num_clusters = np.argmax(scores) + values[0]
print("\nOptimal number of clusters =", num_clusters)

plt.show()






