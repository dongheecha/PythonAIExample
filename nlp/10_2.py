# 어간추출하기

from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer

input_words = ['writing', 'calves', 'be' 'branded', 'horse', 'randomize', 'possibly', 'provision', 'hospital', 'kept',
               'scratchy', 'code']

# 다양한 스테머 객체 만들기
porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

# 스테머 이름별로 포맷에 맞춰 문자열로 저장하고 출력
stemmer_names = ["PORTER", "LANCASTER", "SNOWBALL"]
formatted_text = '{:>16}' * (len(stemmer_names) + 1)
print(
    "\n", formatted_text.format('INPUT WORD', *stemmer_names), "\n",
    '='*68
)

# 입력 단어별로 어간을 추출해 출력
for word in input_words :
    output = [word, porter.stem(word), lancaster.stem(word), snowball.stem(word)]
    print(formatted_text.format(*output))



