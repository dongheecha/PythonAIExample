# 문장 토큰, 단어 토큰하기

from nltk.tokenize import sent_tokenize, word_tokenize, WordPunctTokenizer

input_text = "Do you know how tokenization works? It's actually quite interesting ! Let's analyze a couple of" \
             "sentences and figure it out."

# 문장 단위 토큰화
print("\n Sentence tokenizer:")
print(sent_tokenize(input_text))

# 단어 토큰화
print("\nWord tokenizer:")
print(word_tokenize(input_text))

# WordPunct 토큰하기
print("\nWord punct tokenizer:")
print(WordPunctTokenizer().tokenize(input_text))



