from PIL import Image
import os


input_folder = "data/knn_image_classifier/apples_jpg"
output_folder = "data/knn_image_classifier/apples"


def white_bg_square(img):
    "return a white-background-color image having the img in exact center"
    size = (max(img.size),)*2
    layer = Image.new('RGB', size, (255, 255, 255))
    layer.paste(img, tuple(map(lambda x: int((x[0]-x[1])/2), zip(size, img.size))))
    return layer


images = [(os.path.join(input_folder, image), image) for image in os.listdir(input_folder)]
for full_image, filename in images:
    img = Image.open(full_image)
    square_one = white_bg_square(img)
    square_one.resize((100, 100), Image.ANTIALIAS)
    square_one.save(os.path.join(output_folder, filename))
