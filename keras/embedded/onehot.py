#! /usr/bin/python3
import numpy as np

# サンプル(複数文)
samples = ["the cat is black","the doc is happy"]
token_index = {}
for sample in samples:
    for word in sample.split():
        # 辞書に入ってなものだけ登録
        if word not in token_index:
            # token_index は 1-origin (伝統的にこうするのか？)
            # ちょっと気持ち悪いぞ...
            token_index[word] = len(token_index)+1

# サンプル毎に最初の max_len だけ考慮する
# あまりに長い文は無視する
max_length = 3
print(token_index)

# 結果のTensorを準備する
# results(サンプル数,max_len, token_index+1次元)
print(max(token_index.values()))
ret = np.zeros((len(samples),max_length,max(token_index.values())+1))
#print(ret)
#print(ret.shape)
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        ret[i, j, index] = 1.0
        
#print(ret)
