"""
    使用生成模型补充类别补平衡数据，在补充数据总量一定的情况下，通过nsga2找到最佳分配比例
    然后在该比例下补充后训练模型
    本代码不区分label的差异
"""

from __future__ import print_function, division
import keras
from keras.datasets import mnist, cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.models import load_model
import matplotlib.pyplot as plt
import nsga_two
import random

import os
import numpy as np

# initial初始化设置
# 输入训练图片的shape
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
# 类别数量和噪声维度
num_classes = 10
latent_dim = 100
# 优化器
optimizer_gen = Adam(0.0002, 0.5)
optimizer_dis = Adam(0.0002, 0.5)
# client总数
num_client = 10
# client数据设置
num_large = 800
num_small = 300
num_obj = num_large * 3 + num_small * 7
num_aug_per_label = num_large - num_small


def get_model():
    # 判别模型
    losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']
    discriminator = build_discriminator()
    discriminator.compile(loss=losses,
                          optimizer=optimizer_dis,
                          metrics=['accuracy'])

    # 生成模型
    generator = build_generator()

    # conbine是生成模型和判别模型的结合
    # 判别模型的trainable为False
    # 用于训练生成模型
    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,))
    img = generator([noise, label])
    discriminator.trainable = False
    valid, target_label = discriminator(img)

    combined = Model([noise, label], [valid, target_label])
    combined.compile(loss=losses,
                     optimizer=optimizer_gen)

    return discriminator, generator, combined


def prepare_data():
    # 载入数据库
    data_path = os.path.join(os.getcwd(), "datasets", "mnist.npz")
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path=data_path)

    # 归一化
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_train = np.expand_dims(x_train, axis=3)
    y_train = y_train.reshape(-1, 1)
    x_test = (x_test.astype(np.float32) - 127.5) / 127.5
    x_test = np.expand_dims(x_test, axis=3)
    y_test = y_test.reshape(-1, 1)

    # 将数据按照y的标签放入二维数组classes中
    classes = [[], [], [], [], [], [], [], [], [], []]
    for i in range(60000):
        classes[int(y_train[i])].append(x_train[i])

    # 生成用于存放训练集的list
    xtrain, ytrain = [[] for _ in range(num_client)], [[] for _ in range(num_client)]
    # 将数据分为10组，放入X_train和Y_train中
    index = [0] * 10
    for i in range(num_client):
        for label in range(num_classes):
            if label == i or (label + 1) % num_classes == i or (label + 2) % num_classes == i:
                xtrain[i] += classes[label][index[label]:index[label] + num_large]
                ytrain[i] += [label for _ in range(num_large)]
                index[label] = index[label] + num_large
            else:
                xtrain[i] += classes[label][index[label]:index[label] + num_small]
                ytrain[i] += [label for _ in range(num_small)]
                index[label] = index[label] + num_small

    return xtrain, ytrain, x_test, y_test


def build_generator():
    model = Sequential()

    model.add(Dense(128 * 7 * 7, activation="relu", input_dim=latent_dim))
    model.add(Reshape((7, 7, 128)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(channels, kernel_size=3, padding='same'))
    model.add(Activation("tanh"))

    # model.summary()

    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(num_classes, latent_dim)(label))

    model_input = multiply([noise, label_embedding])
    img = model(model_input)

    return Model([noise, label], img)


def build_discriminator():
    model = Sequential()

    model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    # model.summary()

    img = Input(shape=img_shape)

    # Extract feature representation
    features = model(img)

    # Determine validity and label of the image
    validity = Dense(1, activation="sigmoid")(features)
    label = Dense(num_classes, activation="softmax")(features)

    return Model(img, [validity, label])


# 卷积模型的模板
def getclassifier():
    model = Sequential()
    model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Flatten())  # 平展层，结果输入全连接层
    model.add(Dense(64))  # 全连接层
    model.add(Activation('relu'))
    model.add(Dense(num_classes))  # 全连接层 10个神经元
    model.add(Activation('softmax'))  # 激活函数，公式为y = log(1 + e^x)

    opt = keras.optimizers.RMSprop(lr=0.001)  # 学习率0.001，学习率的含义：在梯度下降法中，学习率是下降步长λ

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['sparse_categorical_accuracy'])
    return model


def nsga2(x_train, y_train, x_test, y_test):
    # 获得一个初始的模型
    num_obj2 = [0] * num_client
    generator, discriminator = fedgan(x_train, y_train, num_obj2)

    # GAN模型保存位置
    gen_path = os.path.join(os.getcwd(), "models", "nsga2", "generator")
    discri_path = os.path.join(os.getcwd(), "models", "nsga2", "discriminator")
    if not os.path.isdir(gen_path):
        os.makedirs(gen_path)
    if not os.path.isdir(discri_path):
        os.makedirs(discri_path)

    # 开始循环
    for nsga_turn in range(3):  # 遗传算法搜索3次
        # 保存本轮用于补充数据的GAN模型
        model_suffix = "nsga_turn" + str(nsga_turn)
        generator.save(os.path.join(gen_path, model_suffix))
        discriminator.save(os.path.join(discri_path, model_suffix))

        # 用当前G模型生成需要补充数据的总量，放入aug_imgs_list
        aug_imgs_list = []
        for label in range(num_classes):
            temp_labels = [label] * num_aug_per_label
            temp_labels = np.array(temp_labels).reshape((num_aug_per_label, 1))
            noise = np.random.normal(0, 1, (num_aug_per_label, latent_dim))
            aug_imgs = generator.predict([noise, temp_labels])
            aug_imgs_list.append(aug_imgs)

        # 初始化一些分布方案
        solu_shape = [12, num_client]  # solutions的shape
        solutions = []
        for i in range(solu_shape[0]):
            solu_temp = []
            for j in range(solu_shape[1]):
                solu_temp.append(random.random())
            solutions.append(solu_temp)

        solutions = np.array(solutions)

        best = 0  # 最佳方案
        max_turn = 20
        current_turn = 0
        for solu_turn in range(max_turn):  # 一次遗传算法迭代次数
            current_turn = solu_turn
            acc_list = []
            for i in range(solu_shape[0]):
                sum = np.sum(solutions[i])
                solutions[i] = solutions[i] / sum

            for solu_i in range(solu_shape[0]):
                # 根据分布把数据给各个client,并更新num_obj2
                start = 0
                for client_i in range(num_client):
                    current_aug_client_i = 0
                    for label in range(num_classes):
                        client_i_label = int(solutions[solu_i][client_i] * num_aug_per_label)
                        x_train[client_i][num_obj + current_aug_client_i:] = aug_imgs_list[label][
                                                                             start:start + client_i_label]
                        aug_label = [label] * client_i_label
                        y_train[client_i][num_obj + current_aug_client_i:] = aug_label[0:]
                        current_aug_client_i = current_aug_client_i + client_i_label  # 已经补充的数据总量(所有label)
                    start = start + client_i_label  # 下一个client对应补充数据的起始位置
                    num_obj2[client_i] = current_aug_client_i

                # sample_images(generator, model_suffix)

                classifier = fedclassfier(x_train, y_train, solu_turn, solu_i)

                # 计算此分布下的discriminator判别各类别的acc
                scores = classifier.evaluate(x_test, y_test, verbose=0)
                acc_list.append(scores[1])

            print("当前一轮迭代完成，迭代次数为：", current_turn)
            print("acc列表：", acc_list)
            # 调用NSGA2算法，交叉重组变异，获得一批新的分布方案以及最佳方案序号
            solutions, best = nsga_two.evolution(solutions, acc_list)

            print("nsga2算法执行完成")

        # 输出最佳分配
        solution = solutions[0]
        print("本轮最优分配策略是：", best)
        print("本轮最优分配策略是：", solution)

        # 联邦训练GAN模型
        generator, discriminator = fedgan(x_train, y_train, num_obj2, havemodel=True)

    model_suffix = "final"
    generator.save(os.path.join(gen_path, model_suffix))
    discriminator.save(os.path.join(discri_path, model_suffix))

# 联邦训练GAN
def fedgan(x_train, y_train, num_obj2, havemodel=False):
    batch_size = 100
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    discri_list = []
    gen_list = []
    com_list = []
    total_turn = 8000

    for i in range(num_client):
        discriminator, generator, combined = get_model()
        discri_list.append(discriminator)
        gen_list.append(generator)
        com_list.append(combined)
    if havemodel:
        total_turn = 2000
        gen_path = os.path.join(os.getcwd(), "models", "TrainAndGen", "nsga2", "generator")
        discri_path = os.path.join(os.getcwd(), "models", "TrainAndGen", "nsga2", "discriminator")
        for i in range(num_client):
            gen_list[i].set_weights(load_model(gen_path).get_weights())
            discri_list[i].set_weights(load_model(discri_path).get_weights())

    loss_threshold = 0.001
    for epoch in range(total_turn):
        d_loss1_total = 0.
        d_loss3_total = 0.
        g_loss_total = 0.
        d_loss2_total = 0.
        d_loss4_total = 0.

        # 每个client单独训练
        for i in range(num_client):
            # 本地训练40轮(一个epoch)
            # for _ in range(10):
            # --------------------- #
            #  sample真实数据
            # --------------------- #
            idx = np.random.randint(0, num_obj + num_obj2[i], batch_size)
            imgs, labels = np.array(x_train[i])[idx], np.array(y_train[i])[idx]

            # ---------------------- #
            #   生成正态分布的输入(更新D模型）
            # ---------------------- #
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            sampled_labels = np.random.randint(0, num_classes, (batch_size, 1))
            gen_imgs = gen_list[i].predict([noise, sampled_labels])

            # --------------------- #
            #  训练D模型
            # --------------------- #
            d_loss_real = discri_list[i].train_on_batch(imgs, [valid, labels])
            d_loss_fake = discri_list[i].train_on_batch(gen_imgs, [fake, sampled_labels])

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # --------------------- #
            #  训练生成模型
            # --------------------- #
            for _ in range(3):
                noise1 = np.random.normal(0, 1, (batch_size, latent_dim))
                sampled_labels1 = np.random.randint(0, num_classes, (batch_size, 1))
                g_loss = com_list[i].train_on_batch([noise1, sampled_labels1], [valid, sampled_labels1])

            # --------------------- #
            #  计算loss和acc
            # --------------------- #
            d_loss1_total += d_loss[1] / num_client
            d_loss2_total += d_loss[2] / num_client
            d_loss3_total += d_loss[3] / num_client
            d_loss4_total += d_loss[4] / num_client
            g_loss_total += g_loss[0] / num_client

        if epoch % 100 == 99:
            print("%d [D loss1: %f, loss2: %f, acc1: %.2f%%, acc2: %.2f%%] [G loss: %f]" % (
                epoch, d_loss1_total, d_loss2_total, d_loss3_total * 100, d_loss4_total * 100, g_loss_total))

        # 用FedAvg方法来做Merge
        gen_ws = []
        for j, lw in enumerate(gen_list[0].get_weights()):
            temp = np.zeros(lw.shape)
            for idd in range(num_client):
                temp = gen_list[idd].get_weights()[j] / num_client + temp
            gen_ws.append(temp)
        for _ in range(num_client):
            gen_list[_].set_weights(gen_ws)

        discri_ws = []
        for j, lw in enumerate(discri_list[0].get_weights()):
            temp = np.zeros(lw.shape)
            for idd in range(num_client):
                temp = discri_list[idd].get_weights()[j] / num_client + temp
            discri_ws.append(temp)
        for _ in range(num_client):
            discri_list[_].set_weights(discri_ws)

        if d_loss1_total <= loss_threshold and g_loss_total <= loss_threshold:
            break

    # if havemodel == False:
    #     sample_images(gen_list[0], str(0))

    return gen_list[0], discri_list[0]


# 联邦训练分类器
def fedclassfier(X_train, Y_train, solu_turn, solu_i):
    maxround = 50
    batch_size = 100
    epochs = 3
    # data_augmentation = False
    save_dir = os.path.join(os.getcwd(), 'models', 'nsga2', 'classifier')
    model_name = 'solu_turn_' + str(solu_turn) + 'solu_i_' + str(solu_i) + '.h5'

    classifier_list, num, snum = [], [], 0
    # 创建多个相同权重的模型，放入mlist中
    imodel = getclassifier()
    for i in range(num_client):
        num.append(X_train[i].shape[0])
        model = getclassifier()
        model.set_weights(imodel.get_weights())
        classifier_list.append(model)
        snum += X_train[i].shape[0]
    print(snum, num)

    for r in range(maxround):
        # mlist中的每个模型，依次使用各个训练集进行训练
        for i in range(num_client):
            x_train, y_train = X_train[i], Y_train[i]
            # print('Not using data augmentation.')
            classifier_list[i].fit(x_train, y_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         verbose=0,
                         shuffle=True)

        # 对mlist中各模型的weights进行加权求和，放入列表ws中
        ws = []
        for j, lw in enumerate(classifier_list[0].get_weights()):
            temp = np.zeros(lw.shape)
            for idd in range(num_client):
                temp += num[idd] / snum * classifier_list[idd].get_weights()[j]
            ws.append(temp)
        for i in range(num_client):
            classifier_list[i].set_weights(ws)

    # 循环结束，保存classfier，将每次循环的损失值以及准确值保存到对应的npy文件中
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    classifier_list[0].save(model_path)
    print('Saved trained classifier model at %s ' % model_path)

    return classifier_list[0]

# def sample_images(generator, str):
#     r, c = 10, 10
#     noise = np.random.normal(0, 1, (r * c, latent_dim))
#     sampled_labels = np.array([num for _ in range(r) for num in range(c)])
#     gen_imgs = generator.predict([noise, sampled_labels])
#     # Rescale images 0 - 1
#     gen_imgs = 0.5 * gen_imgs + 0.5
#
#     fig, axs = plt.subplots(r, c)
#     cnt = 0
#     for i in range(r):
#         for j in range(c):
#             axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
#             axs[i, j].axis('off')
#             cnt += 1
#     fig.savefig("images/images_tandg/nsga2/%s.png" % str)
#     plt.close()


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = prepare_data()
    nsga2(X_train, Y_train, X_test, Y_test)
