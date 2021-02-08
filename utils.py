from PIL import Image


def show_img(np_array):
    img = Image.fromarray(np_array, 'RGB')
    img.show()

def save_img(np_array, path):
    img = Image.fromarray(np_array, 'RGB')
    img.save(path)