import argparse
import simpleai.search as ss

class CustomProblem(ss.SearchProblem):
    def set_target(self, target_string):
        self.target_string = target_string

    # 현재 상태를 확인하고 적절한 액션 취하기
    def actions(self, cur_state):
        if len(cur_state) < len(self.target_string):
            alphabets = 'abcdefghijklmnopqrstuvwxyz'
            return list(alphabets + ' ' + alphabets.upper())
        else:
            return []
        
    # cur_state와 action 문자열을 결합해 결과 만들기
    def result(self, cur_state, action):
        return cur_state + action

    # 목표에 도달했는지 확인하기
    def is_goal(self, cur_state):
        return cur_state == self.target_string
    
    # 예제에서 사용할 휴리스틱 정의하디
    def heuristic(self, cur_state):
        # 현재 문자열과 목표 문자열 비교하기
        dist = sum([1 if cur_state[i] != self.target_string[i] else 0 for i in range(len(cur_state))])
        
        #두 문자열의 길이 차이 비교하기
        diff = len(self.target_string) - len(cur_state)

        return dist + diff

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Creates the input\ string using the greedy algorithm')
    parser.add_argument("--input-string", dest="input_string", required=True, help="Input string")
    parser.add_argument("--initial-state", dest="initial_state", required=False, default='', help="Starting point"
                                                                                                  "for the search")
    return parser

if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    
    # 오브젝트 초기화
    problem = CustomProblem()
    
    # 목표 문자열과 초기상태 설정하기
    problem.set_target(args.input_string)
    problem.initial_state = args.initial_state

    #문제 풀기
    output = ss.greedy(problem)

    print('\nTarget string:', args.input_string)
    print('\nPath to the solution:')
    for item in output.path():
        print(item)




