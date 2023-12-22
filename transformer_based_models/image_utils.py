from moviepy.video.io.bindings import mplfig_to_npimage
from pathlib import Path
from PIL import Image


def _figure_to_array(fig):
    arr = mplfig_to_npimage(fig)
    return arr


def _to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def save_image(img, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    _to_pil(img).save(str(path))


def show_image(img):
    copied = img.copy()
    copied = _to_pil(copied)
    copied.show()


def _figure_to_array(fig):
    arr = mplfig_to_npimage(fig)
    return arr
