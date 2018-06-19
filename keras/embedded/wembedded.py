#! /usr/bin/python3
from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding

# 特徴量として考慮する単語の数
max_features = 100000

# max_featues 個の出現頻度の高い語彙
# のうち先頭から以下の数を残してカット
# 短いものは0パディング
max_len = 100

# データを読み込みます
(x_train, y_train),(x_test, y_test) = imdb.load_data(num_words=max_features)

# 各データのTensorの形状を確認
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# 整数のリストが (samples, max_lken) の整数値の2次元Tensorになるように変換する
x_train = preprocessing.sequence.pad_sequences(x_train, max_len)
x_test  = preprocessing.sequence.pad_sequences(x_test, max_len)

print(x_train.shape)

# モデルの定義
model = Sequential()
# Embedding
# input shape:  (batch_size,sequence_length)
# output shape: (batch_size,sequence_length, output_dim)
model.add(Embedding(max_features, 128, input_length=max_len))
# Flattern()で (batch_size, sequence_length x output_dim) に変換 
model.add(Flatten())

# 分類器
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics = ["acc"])
model.summary()

# 学習(validation_split は trainingデータの一部をacc計算に利用する)
history = model.fit(x_train,
                    y_train,
                    epochs = 200,
                    batch_size = 32,
                    validation_split=0.2)
