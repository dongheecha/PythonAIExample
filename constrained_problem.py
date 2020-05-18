from simpleai.search import \
    (CspProblem, backtrack, min_conflicts, MOST_CONSTRAINED_VARIABLE, HIGHEST_DEGREE_VARIABLE, LEAST_CONSTRAINING_VALUE)

# 변수마다 고유한 값을 가져야 한다는 제약 조건
def constraint_unique(variables, values):
    # 서로 값이 다른지 검사하기
    return len(values) == len(set(values))