import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. 加载CIFAR-10数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# 数据预处理
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 标签名称
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 1. 展示任意一张图片像素
print("步骤1: 展示任意一张图片像素")
print("第一张训练图像的像素值形状:", x_train[0].shape)
print("像素值范围: [{:.3f}, {:.3f}]".format(x_train[0].min(), x_train[0].max()))
print("对应标签:", labels[y_train[0][0]])

# 2. 展示训练集的前10张图像
print("\n步骤2: 展示训练集的前10张图像")
plt.figure(figsize=(15, 3))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_train[i])
    plt.title(labels[y_train[i][0]])
    plt.axis('off')
plt.tight_layout()
plt.show()

# 3. 根据模型图构建模型
print("\n步骤3: 构建CNN模型")
K = len(labels)  # 类别数量

# 使用Functional API构建模型
# 输入层
i = tf.keras.layers.Input(shape=(32, 32, 3))

# 第一卷积层
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(i)
x = tf.keras.layers.BatchNormalization()(x)

# 第二卷积层
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)

# 最大池化层和Dropout
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Dropout(0.25)(x)

# 展平层
x = tf.keras.layers.Flatten()(x)

# 全连接层
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)

# 输出层
x = tf.keras.layers.Dense(K, activation='softmax')(x)

# 创建模型
model = tf.keras.Model(i, x)

# 显示模型结构
model.summary()

# 4. 设置优化器和评估指标
print("\n步骤4: 编译模型")
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. 使用TensorFlow方法划分验证集
print("\n步骤5: 训练模型（使用TensorFlow划分验证集）")

# 使用TensorFlow的validation_split参数自动划分验证集
# validation_split=0.2 表示使用20%的训练数据作为验证集
history = model.fit(x_train, y_train,
                    batch_size=32,
                    epochs=2,
                    validation_split=0.2,  # 使用20%的训练数据作为验证集
                    shuffle=True,  # 打乱数据
                    verbose=1)

# 6. 展示训练精度和验证精度的曲线
print("\n步骤6: 展示训练曲线")
plt.figure(figsize=(12, 4))

# 准确率曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 损失曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 7. 评估测试集精度
print("\n步骤7: 评估测试集精度")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"测试集损失: {test_loss:.4f}")
print(f"测试集准确率: {test_accuracy:.4f}")

# 8. 展示测试集指定图像的预测结果和10种类别的概率
print("\n步骤8: 展示测试集图像的预测结果")

# 选择测试集中的一些图像进行预测
sample_indices = [0, 1, 2, 3, 4]  # 可以选择不同的索引

plt.figure(figsize=(15, 10))
for i, idx in enumerate(sample_indices):
    # 获取图像和真实标签
    image = x_test[idx]
    true_label = y_test[idx][0]

    # 预测
    prediction = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
    predicted_label = np.argmax(prediction)

    # 显示图像
    plt.subplot(2, 5, i + 1)
    plt.imshow(image)
    plt.title(f'True: {labels[true_label]}\nPred: {labels[predicted_label]}')
    plt.axis('off')

    # 显示概率分布
    plt.subplot(2, 5, i + 6)
    plt.barh(labels, prediction)
    plt.xlabel('Probability')
    plt.title('Class Probabilities')
    plt.xlim(0, 1)

plt.tight_layout()
plt.show()

# 额外：展示一个详细预测示例
print("\n详细预测示例:")
sample_idx = 0
sample_image = x_test[sample_idx]
true_label_idx = y_test[sample_idx][0]

# 预测
prediction = model.predict(np.expand_dims(sample_image, axis=0), verbose=0)[0]
predicted_label_idx = np.argmax(prediction)

print(f"真实标签: {labels[true_label_idx]}")
print(f"预测标签: {labels[predicted_label_idx]}")
print(f"预测置信度: {prediction[predicted_label_idx]:.4f}")

print("\n所有类别的概率:")
for i, (label, prob) in enumerate(zip(labels, prediction)):
    print(f"{label}: {prob:.4f}")