#! /usr/bin/python3
#
# 【cnn_cifar10】
#
#  概要: CIFAR-10の画像をCNNで学習/推定
#
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dense

# 学習データのロード
(x_train, y_train),(x_test, y_text) = cifar10.load_data()

# 学習データの正規化
x_train = x_train / 255.0
x_test = x_test / 255.0

# 1-hot ベクトルの作成
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_text,10)

# モデルの作成
model = Sequential()

# モデルの作成
model.add(
    Conv2D(
        # フィルタの次元
        filters = 32,
        input_shape = (32,32,3),
        kernel_size = (3,3),
        strides = (1,1),
        padding = "same",
        activation = "relu",
    )
)
model.add(
    Conv2D(
        filters = 32,
        kernel_size = (3,3),
        strides = (1,1),
        padding = "same",
        activation = "relu",
    )
)
model.add(
    MaxPooling2D(
        # 画像サイズが1/2 になる
        # 16
        pool_size=(2,2),
    )
)
model.add(
    Dropout(0.25),
)

model.add(
    Conv2D(
        filters=64,
        kernel_size = (3,3),
        strides=(1,1),
        padding = "same",
        activation = "relu",
    )
)

model.add(
    Conv2D(
        filters = 64,
        kernel_size = (3,3),
        strides=(1,1),
        padding = "same",
        activation = "relu",
    )
)

model.add(
    MaxPooling2D(
        # 画像サイズが1/2 になる
        # 8
        pool_size = (2,2)
    )
)

model.add(Dropout(0.25))
# (None, 8, 8, 64)
print("model size: {}".format(model.output_shape))

model.add(Flatten())
print("model size: {}".format(model.output_shape))
model.add(
    Dense(
        units=512,
        activation = "relu",
    )
)
print("model size: {}".format(model.output_shape))
model.add(Dropout(0.5))
model.add(
    Dense(
        units=10,
        activation = "softmax",
    )
)
print("model size: {}".format(model.output_shape))

# モデルをコンパイル
model.compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = ["accuracy"],
)

hist = model.fit(
    x_train,
    y_train,
    batch_size = 32,
    epochs = 500,
    validation_split = 0.2,
)
