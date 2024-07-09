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

# Định nghĩa thư mục data
data_folder = "dataset"

# Khởi tạo model
model = extract_feature_model()

vectors = []
paths = []

for image_path in os.listdir(data_folder):

    # Nối full path
    image_path_full = os.path.join(data_folder, image_path)

    # Trích chọn đặc trưng
    image_vector = extract_vector(model, image_path_full)

    # Add các đặc trưng và full path vào list
    vectors.append(image_vector)
    paths.append(image_path_full)

# Save file
vector_file = "vectors.pkl"
path_file = "paths.pkl"

pickle.dump(vectors, open(vector_file, "wb"))
pickle.dump(paths, open(path_file, "wb"))

