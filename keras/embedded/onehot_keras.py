#! /usr/bin/python3
from keras.preprocessing.text import Tokenizer
# サンプル(複数文)
samples = ["the cat is black","the doc is happy"]
# 出現頻度がもっとも高い1000個の単語だけを抽出する
tokenizer = Tokenizer(num_words=10)
tokenizer.fit_on_texts(samples)
print(tokenizer)
# id 番号の文字列配列に変換(便利)
sequences = tokenizer.texts_to_sequences(samples)
print(sequences)

one_hot_results = tokenizer.texts_to_matrix(samples, mode="binary")
for sample in one_hot_results:
    # (num_words,) で binary がたっているベクトルに変換される
    print(sample.shape)

print(tokenizer.word_index)

