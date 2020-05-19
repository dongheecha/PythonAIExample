import numpy as np
from nltk.corpus import brown

# 입력 테스트를 단어 묶음으로 나눈다.
# 단어 묶음별로 N개의 단어를 포함한다
def chunker(input_data, N) :
    input_words = input_data.split(' ')
    output = []

    cur_chunk = []
    count = 0
    for word in input_words:
        cur_chunk.append(word)
        count += 1
        if count == N:
            output.append(' '.join(cur_chunk))
            count, cur_chunk = 0, []

    output.append(' '.join(cur_chunk))
    return output

if __name__ == '__main__':
    # 브라운 코퍼스에서 첫 12,000개 단어를 읽어온다.
    input_data = ' '.join(brown.words()[:12000])
    
    #각 단어 묶음에 포함될 단어 수 정의
    chunk_size = 700

    chunks = chunker(input_data, chunk_size)
    print('\nNumber of text chunks =', len(chunks), '\n')
    for i, chunk in enumerate(chunks):
        print('Chunk', i+1, '-->', chunk[:50])
