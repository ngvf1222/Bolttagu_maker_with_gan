import os
import time
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 데이터 전처리
data_list = os.listdir("image")
datas = [cv2.imread("image/" + i) for i in data_list]
datas = np.array([cv2.resize(i, (128, 128)) for i in datas])
datas = (datas.astype("float64") - 127.5) / 127.5
print(len(datas))
# 변수 설정
BUFFER_SIZE = 60000
BATCH_SIZE = 4

# 데이터 배치를 만들고 섞습니다.
train_dataset = (
    tf.data.Dataset.from_tensor_slices(datas).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
)

def show(img):
    float_img = (img + 1) * 127.5
    color_img=np.floor(float_img).astype("uint8")[0]
    #print(color_img)
    cv2.imshow("img", color_img)
    cv2.waitKey()
def processing_result(img):
    float_img = (np.clip(img,-1,1) + 1) * 127.5
    return np.floor(float_img).astype("uint8")[0]
def save(img, name):
    float_img = (np.clip(img,-1,1) + 1) * 127.5
    cv2.imwrite(f"./test/{name}.png", np.floor(float_img).astype("uint8")[0])
    print(name)


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(32 * 32 * 256, use_bias=False, input_shape=(128,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((32, 32, 256)))
    assert model.output_shape == (
        None,
        32,
        32,
        256,
    )  # 주목: 배치사이즈로 None이 주어집니다.

    model.add(
        tf.keras.layers.Conv2DTranspose(
            128, (5, 5), strides=(1, 1), padding="same", use_bias=False
        )
    )
    assert model.output_shape == (None, 32, 32, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(
        tf.keras.layers.Conv2DTranspose(
            64, (5, 5), strides=(2, 2), padding="same", use_bias=False
        )
    )
    assert model.output_shape == (None, 64, 64, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(
        tf.keras.layers.Conv2DTranspose(
            3, (5, 5), strides=(2, 2), padding="same", use_bias=False, activation="tanh"
        )
    )
    assert model.output_shape == (None, 128, 128, 3)

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2D(
            64, (5, 5), strides=(2, 2), padding="same", input_shape=[128, 128, 3]
        )
    )
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model


generator = make_generator_model()
discriminator = make_discriminator_model()
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
checkpoint_dir = "./training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator,
)
try:
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
except Exception:
    pass
finally:
    for i in range(100):
        noise=tf.random.normal([1, 128])
        save(np.array(generator(noise, training=False)),'{i}')
