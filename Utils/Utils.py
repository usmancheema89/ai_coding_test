from PIL import Image
import numpy as np

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # print(path)
    with open(path, 'rb') as f:
        img = Image.open(f).convert('RGBA')
        x = np.array(img)
        r, g, b, a = np.rollaxis(x, axis = -1)
        r[a == 0] = 255
        g[a == 0] = 255
        b[a == 0] = 255
        x = np.dstack([r, g, b, a])
        img = Image.fromarray(x, 'RGBA')
        return img.convert('RGB')
