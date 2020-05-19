# 어간추출하기

from nltk.stem import WordNetLemmatizer

input_words = ['writing', 'calves', 'be' 'branded', 'horse', 'randomize', 'possibly', 'provision', 'hospital', 'kept',
               'scratchy', 'code']

# 표제화기 객체 생성
lemmatizer = WordNetLemmatizer()

# 표제화기 이름별로 포맷에 맞춰 문자열로 저장하고 출력
lemmatizer_names = ['NOUN LEMMATIZER', 'VERB LEMMATIZER']
formatted_text = '{:>24}' * (len(lemmatizer_names) + 1)
print('\n', formatted_text.format('INPUT WORD', *lemmatizer_names), '\n', '='*75)

# 입력단어를 표제화하고 결과를 출력한다.
for word in input_words :
    output = [word, lemmatizer.lemmatize(word, pos='n'), lemmatizer.lemmatize(word, pos='v')]
    print(formatted_text.format(*output))

