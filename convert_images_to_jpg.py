import os
from PIL import Image

path = "data/knn_image_classifier/apples_original"
image_paths = [os.path.join(path, image) for image in os.listdir(path)]

for image in image_paths:
    png = Image.open(image).convert('RGB')
    background = Image.new('RGBA', png.size, (255, 255, 255))

    # alpha_composite = Image.alpha_composite(background, png)
    png.save(f'data/knn_image_classifier/apples_jpg/{"".join(os.path.basename(image).split(".")[:-1])}.jpg', 'JPEG', quality=80)
