import math
import os

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

from PIL import Image
import pickle
import numpy as np

# Hàm tạo model
def extract_feature_model():
    vgg16_model = VGG16(weights="imagenet")
    extract_model = Model(inputs=vgg16_model.inputs, outputs=vgg16_model.get_layer("fc1").output)
    return extract_model

# Hàm tiền xử lý, chuyển đổi ảnh thành tensor
def image_preprocessing(img):
    img = img.resize((224, 224))
    img = img.convert("RGB")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def extract_vector(model, image_path):
    print("Đang xử lý: ", image_path)
    img = Image.open(image_path)
    img_tensor = image_preprocessing(img)

    # Trích chọn đặc trưng
    vector = model.predict(img_tensor)[0]

    # Chuẩn hóa vector (bằng cách chia cho L2 norm, vẫn giữ nguyên được hướng)
    # L2 norm: L2 norm của một vector là căn bậc hai của tổng các bình phương các phần tử của vector đó. Nó còn được gọi là Euclidean norm.
    # Chuẩn hóa L2: Để chuẩn hóa một vector bằng L2 norm, chúng ta chia mỗi phần tử của vector đó cho L2 norm của chính nó.

    vector = vector / np.linalg.norm(vector)
    return vector

# Định nghĩa ảnh cần tìm kiếm
search_image = "dataset\999.jpg"

# Khởi tạo model
model = extract_feature_model()

# Trích chọn đặc trưng
search_vector = extract_vector(model, search_image)

# Load danh sách các vector trong file vectors.pkl ra biến
vectors = pickle.load(open("vectors.pkl", "rb"))
paths = pickle.load(open("paths.pkl", "rb"))

# Tính khoảng cách từ search_vector đến tất cả các vector ảnh còn lại
distance = np.linalg.norm(vectors - search_vector, axis=1)

# Sắp xếp và lấy ra k vector có khoảng cách ngắn nhất (gần giống với ảnh search nhất)
k = 16
ids = np.argsort(distance)[:k] # Sắp xếp mảng distance theo thứ tự tăng dần, sau đó chỉ lấy ra id của ảnh

# Tạo output
nearest_image = [(paths[id], distance[id]) for id in ids]

# Vẽ lên màn hình các ảnh gần nhất đó
import matplotlib.pyplot as plt

axes = []
grid_size = int(math.sqrt(k))
fig = plt.figure(figsize=(10, 5))

for id in range(k):
    draw_image = nearest_image[id]
    axes.append(fig.add_subplot(grid_size, grid_size, id+1))

    axes[-1].set_title(draw_image[1])
    plt.imshow(Image.open(draw_image[0]))

fig.tight_layout()
plt.show()

### FAISS Library