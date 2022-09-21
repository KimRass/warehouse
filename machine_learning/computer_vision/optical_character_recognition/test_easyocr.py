import easyocr
from PIL import Image, ImageDraw
from pathlib import Path


def draw_bboxes(image, bounds, color="red", width=2):
    draw = ImageDraw.Draw(image)

    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)

        conf = bound[2]
        if conf >= 0.8:
            print(f"{bound[1]:100s} ({bound[2]:.1%})")
    return image


dir = Path("/Users/jongbeom.kim/Downloads/menu_images")
for img_path in dir.glob("**/*"):
    img = Image.open(img_path)

    # Load a model into memory. It takes some time but it needs to be run only once.
    # The argument is the list of languages you want to read. You can pass several languages at once but not all languages can be used together. English is compatible with every language and languages that share common characters are usually compatible with each other.
    # Instead of the filepath, you can also pass an OpenCV image object (numpy array) or an image file as bytes. A URL to a raw image is also acceptable.
    reader = easyocr.Reader(["ko"])

    # The output will be in a List format, each item represents a bounding box, the text detected and confident level, respectively.
    # `detail=0`: simpler output.
    # `gpu=False`: In case you do not have a GPU, or your GPU has low memory, you can run the model in CPU-only mode.
    bounds = reader.readtext(img)

    img_with_bboxes = draw_bboxes(img, bounds)
    # img_with_bboxes.show()
    img_with_bboxes.save(f"/Users/jongbeom.kim/Downloads/easyocr_outputs/{img_path.stem}_easyocr.png")

