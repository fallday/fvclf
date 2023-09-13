import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 加载已训练的模型
model = load_model('mv3.h5')  # 替换成你保存的模型文件名

# 定义类别名称（根据你的数据集）
# class_names = ['apple', 'banana', 'grapes', 'kiwi', 'mango', 'orange', 'pear', 'pinapple', 'pomegranate', 'watermelon']  # 替换成你的类别名称
class_names = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber',
               'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear',
               'peas', 'pinapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']

# 加载测试数据集
test_dir = 'dataset/test'  # 替换成你的测试数据集路径
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224,224),
    batch_size=1,  # 使用批大小为1来逐个预测
    class_mode='categorical',
    shuffle=False
)

# 随机选择25张测试图片
num_images_to_show = 25
sample_indices = random.sample(range(len(test_generator)), num_images_to_show)

# 创建一个5x5的子图网格
fig, axes = plt.subplots(5, 5, figsize=(20, 20))
axes = axes.ravel()

for i, sample_index in enumerate(sample_indices):
    # 获取测试图像和真实标签
    test_image, true_label = test_generator[sample_index]
    
    # 预测图像类别
    predicted_probs = model.predict(test_image)
    predicted_class_index = np.argmax(predicted_probs)
    predicted_class = class_names[predicted_class_index]
    
    true_class_index = np.argmax(true_label)
    true_class = class_names[true_class_index]
    
    # 将图像从(1, 224, 224, 3)的形状变换为(224, 224, 3)
    test_image = np.squeeze(test_image)
    
    # 绘制图像
    axes[i].imshow(test_image)
    axes[i].set_title(f'P: {predicted_class}, T: {true_class}')
    axes[i].axis('off')

plt.subplots_adjust(wspace=0.5)
plt.show()
