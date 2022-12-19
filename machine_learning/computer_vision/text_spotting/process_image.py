import cv2
import requests
import numpy as np
import matplotlib.pyplot as plt


def load_image_as_array(url_or_path="", gray=False):
    url_or_path = str(url_or_path)

    if "http" in url_or_path:
        img_arr = np.asarray(
            bytearray(requests.get(url_or_path).content), dtype="uint8"
        )
        if not gray:
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)
    else:
        if not gray:
            img = cv2.imread(url_or_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imread(url_or_path, cv2.IMREAD_GRAYSCALE)
    return img


def show_image(img1, img2=None, alpha=0.5):
    plt.figure(figsize=(10, 8))
    plt.imshow(img1)
    if img2 is not None:
        plt.imshow(img2, alpha=alpha)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def draw_rectangles_on_image(img, rectangles1, rectangles2):
    img_copied = img.copy()

    for xmin, ymin, xmax, ymax in rectangles1[["xmin", "ymin", "xmax", "ymax"]].values:
        cv2.rectangle(
            img=img_copied, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(255, 0, 0), thickness=1
        )
    for xmin, ymin, xmax, ymax in rectangles2[["xmin", "ymin", "xmax", "ymax"]].values:
        cv2.rectangle(
            img=img_copied, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(0, 0, 255), thickness=1
        )

    # if "block" in rectangles.columns.tolist():
    #     for block, xmin, ymin, xmax, ymax in rectangles.drop_duplicates(["block"])[
    #         ["block", "xmin", "ymin", "xmax", "ymax"]
    #     ].values:
    #         cv2.putText(
    #             img=img_copied,
    #             text=str(block),
    #             org=(xmin, ymin),
    #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #             fontScale=0.5,
    #             color=(255, 0, 0),
    #             thickness=2
    #         )
    return img_copied