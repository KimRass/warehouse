import easyocr
from PIL import Image, ImageDraw, ImageFont


def draw_bboxes(img_path, color="yellow", width=2):
    # detection="DB", recognition="Transformer"
    reader = easyocr.Reader(["ko", "en"], gpu=False)

    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    results = reader.readtext(img_path)
    for (p0, p1, p2, p3), label, conf in results:
        # print(p0, p1, p2, p3, label)
        size = (p3[1] - p1[1]) // 4
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
        draw.text(
            xy=p0,
            text=f"{label} ({conf:.1%})",
            font=ImageFont.truetype(font="AppleGothic.ttf", size=size),
            # align="left",
            anchor="lb",
            fill=(255, 255, 255)
        )
    return img


img_path = "/Users/jongbeom.kim/Desktop/workspace/data_science/machine_learning/computer_vision/datasets/EasyOCR/korean.png"
img_with_bboxes = draw_bboxes(img_path)
img_with_bboxes.show()


# img = Image.open(img_path)
# draw = ImageDraw.Draw(img)
# font = ImageFont.truetype("AppleGothic.ttf", 36)
# draw.text(xy=(100, 200), text="서울", font=font)
# img.show()