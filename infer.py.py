import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# 加载训练好的模型
model = load_model('fruit_veggie_model.h5')

# 图像路径
image_path = 'dataset/test/pomegranate/Image_2.jpg'

# 加载和预处理图像
img = load_img(image_path, target_size=(224, 224))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# 进行推理
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)

# 打印结果
print("Predicted class:", predicted_class)
