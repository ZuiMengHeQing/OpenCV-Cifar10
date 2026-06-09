"""
卷积解释----->第一卷积层：filter=32，kernel_size=(3,3), input_shape=(32,32,3), activation=relu, padding=same
第 二 卷 积 层 ： filter=64 ， kernel_size=(3,3), activation=relu, padding=same
解释：
filters=32：表示使用32个卷积核（滤波器），每个卷积核会提取输入中的某种特征，因此输出特征图的深度（通道数）为32。
kernel_size=(3,3)：每个卷积核的大小是3x3，即卷积核在输入图像上滑动时每次查看3x3的区域。
input_shape=(32,32,3)：这是输入图像的形状，32x32像素，3个颜色通道（RGB）。注意，在Keras中，如果使用第一层（即输入层），需要指定input_shape，但后续层会自动推断输入形状。
activation='relu'：使用ReLU（Rectified Linear Unit）激活函数，将负值置为零，正值保留。这有助于引入非线性，使网络能够学习更复杂的模式。
padding='same'：在卷积操作时，对输入图像进行填充，使得输出特征图在空间维度（高度和宽度）上与输入相同。具体来说，对于步长为1的卷积，如果使用'same'填充，则输出尺寸等于输入尺寸（不填充）
示例代码：
Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3))
如果不是第一层，则不需要指定input_shape，例如：
x = Conv2D(32, (3,3), activation='relu', padding='same')(x)

最大池化解释----->最大池化层：pool_size=(2,2)，dropout rate: 0.25
最大池化层（MaxPooling2D）
参数：pool_size=(2,2)
含义：
池化窗口的大小为2x2，即在2x2的区域内进行下采样。
ropout层
参数：rate=0.25
含义：
在训练过程中，随机“丢弃”（即暂时忽略）25%的神经元。
"""

# coding: utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
    获取数据
"""
# 加载数据（第一次会从网上下载数据到本地）
cifar10 = tf.keras.datasets.cifar10
# 将其分配到训练集和测试集中
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# (50000, 32, 32, 3) (50000, 1) (10000, 32, 32, 3) (10000, 1)
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = y_train.flatten(), y_test.flatten()
# print(y_train)
# print(y_test)
# plt.subplots绘制5行5列，大小12, 10，一起展示（绘图25个取的是训练级前25个）
fig, ax = plt.subplots(5, 5, figsize=(12, 10))
k = 0
for i in range(5):
    for j in range(5):
        ax[i][j].imshow(x_train[k], aspect='auto')
        k += 1
plt.show()
"""
    模型训练
"""
# 1、 初始化设置
K = len(set(y_train))  # 计算类别数量
print("number of classes:", K)  # 输出类别数（10个）
# 2、模型架构
i = tf.keras.layers.Input(shape=x_train[0].shape)  # 定义输入层，形状与训练数据相同
# print(i)
# 3、第一个卷积块
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(i)  # 32个3x3卷积核
x = tf.keras.layers.BatchNormalization()(x)  # 批标准化，加速训练并提高稳定性
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)  # 再次卷积
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)  # 2x2最大池化，减小特征图尺寸
# 4、第二个卷积块
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)  # 增加至64个卷积核
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
# 5、第三个卷积块
x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)  # 增加至128个卷积核
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
# 6、分类器部分
# 展平和全连接层
x = tf.keras.layers.Flatten()(x)  # 将多维特征图展平为一维向量
x = tf.keras.layers.Dropout(0.2)(x)  # 20%的dropout防止过拟合
x = tf.keras.layers.Dense(1024, activation='relu')(x)  # 全连接层，1024个神经元
x = tf.keras.layers.Dropout(0.2)(x)  # 再次dropout
x = tf.keras.layers.Dense(K, activation='softmax')(x)  # 输出层，使用softmax激活函数
# 7、模型创建
model = tf.keras.Model(i, x)  # 创建模型，指定输入和输出
model.summary()  # 显示模型结构摘要
"""
    我们的模型现在已经准备好了，是时候对它进行编译了(编译模型)。
    我们正在使用model.compile()函数来编译我们的模型。对于参数，我们使用
1、optimizer='adam':
优化器（optimizer）用于更新模型的权重以最小化损失函数。Adam（Adaptive Moment Estimation）是一种常用的优化算法，它结合了动量法和RMSProp的优点。
Adam通过计算梯度的一阶矩估计（均值）和二阶矩估计（未中心化的方差）来调整每个参数的学习率。
优点：自适应学习率，通常收敛速度快，适合大多数问题。
2、loss='sparse_categorical_crossentropy':稀疏分类交叉熵作为损失函数
损失函数（loss）用于衡量模型预测值与真实值之间的差距。sparse_categorical_crossentropy是分类问题中常用的损失函数之一。
这个损失函数适用于整数标签（即标签是0,1,2,...这样的整数）的多分类问题。例如，如果有10个类别，那么标签就是0到9。
它与categorical_crossentropy的区别在于：categorical_crossentropy要求标签是one-hot编码，而sparse_categorical_crossentropy则直接使用整数标签，不需要one-hot编码。
3、metrics=['accuracy']:
评估指标（metrics）用于在训练和测试过程中监控模型的性能。这里使用准确率（accuracy）作为评估指标。
准确率表示模型预测正确的样本数占总样本数的比例。
"""
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
"""
    拟合模型
1、epochs=50
训练轮数：整个训练数据集被使用50次
每个epoch包含多个batch，模型看到所有训练数据一次
"""
# Fit
r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50)

"""
    现在我们已经训练了我们的模型，在对它进行任何预测之前，让我们把每次迭代的准确率可视化，以便更好地分析。
    尽管还有其他的方法，包括混淆矩阵，以更好地分析模型。
"""
plt.plot(r.history['accuracy'], label='acc', color='red')
plt.plot(r.history['val_accuracy'], label='val_acc', color='green')
plt.legend()
"""
    这段代码用于对测试集中的图像进行预测，并展示预测结果与真实标签的对比
"""
labels = '''airplane automobile bird cat deerdog frog horseship truck'''.split() # 所有标签的英文名称
# 选择测试图像下标
image_number = 0
# x_test[image_number] 是形状为(32, 32, 3)的RGB图像
plt.imshow(x_test[image_number])
n = np.array(x_test[image_number])  # 转换为numpy数组
# 重塑原因：模型model期望的输入形状是 (batch_size, 32, 32, 3)从 (32, 32, 3) 变为 (1, 32, 32, 3)，添加了批次维度
p = n.reshape(1, 32, 32, 3)  # 重塑为批量格式
# model.predict(p)返回的是形状为 (1, 10) 的概率数组，.argmax()：找到概率最大的索引，labels[...]：通过索引获取对应的类别名称
predicted_label = labels[model.predict(p).argmax()]
# 获取真实标签
original_label = labels[y_test[image_number]]
print("Original label is {} and predicted label is {}".format(original_label, predicted_label))
"""
    最后，让我们使用model.save()函数将我们的模型保存为h5文件。
"""
# save the model
model.save('geeksforgeeks.h5')
