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


def save(img, name):
    print(img)
    float_img = (np.clip(img,-1,1) + 1) * 127.5
    cv2.imwrite(f"./cache/{name}", np.floor(float_img).astype("uint8"))


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
noise = tf.random.normal([1, 128])
generated_image = generator(noise, training=False)
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

EPOCHS = 1000
noise_dim = 128
num_examples_to_generate = 1

# 이 시드를 시간이 지나도 재활용하겠습니다.
# (GIF 애니메이션에서 진전 내용을 시각화하는데 쉽기 때문입니다.)
seed = tf.random.normal([1, noise_dim])


# `tf.function`이 어떻게 사용되는지 주목해 주세요.
# 이 데코레이터는 함수를 "컴파일"합니다.
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables)
    )
    return gen_loss, disc_loss


def generate_and_save_images(model, epoch, test_input):
    # `training`이 False로 맞춰진 것을 주목하세요.
    # 이렇게 하면 (배치정규화를 포함하여) 모든 층들이 추론 모드로 실행됩니다.
    predictions = model(test_input, training=False)
    save(np.array(predictions[0]), "image_at_epoch_{:04d}.png".format(epoch))


loss = []


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            loss.append(train_step(image_batch))
        generate_and_save_images(generator, epoch + 1, seed)

        # 15 에포크가 지날 때마다 모델을 저장합니다.
        if (epoch + 1) % EPOCHS == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        # print (' 에포크 {} 에서 걸린 시간은 {} 초 입니다'.format(epoch +1, time.time()-start))
        print("Time for epoch {} is {} sec".format(epoch + 1, time.time() - start))
    generate_and_save_images(generator, epochs, seed)


train(train_dataset, EPOCHS)
plt.plot(loss)
