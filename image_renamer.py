import os

input_path = "data/knn_image_classifier/apples_original"
current_image = 1
for image in os.listdir(input_path):
    os.rename(os.path.join(input_path, image), os.path.join(input_path, f"{current_image}.{image.split('.')[-1]}"))
    current_image += 1
