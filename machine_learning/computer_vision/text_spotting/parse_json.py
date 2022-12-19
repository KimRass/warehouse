import json
import pandas as pd
from pathlib import Path

from process_image import (
    load_image_as_array,
    show_image,
    draw_rectangles_on_image
)


def get_image_and_label(path_json):
    path_json = Path(path_json)

    with open(path_json, mode="r") as f:
        label = json.load(f)

    ls_row = list()
    for sample in label["annotations"]:
        text = sample["annotation.text"]
        xmin, ymin, width, height = sample["annotation.bbox"]
        
        ls_row.append([text, xmin, ymin, xmin + width, ymin + height])
    df_label = pd.DataFrame(ls_row, columns=["text", "xmin", "ymin", "xmax", "ymax"])

    path_img = f"{path_json.parents[4]}/원천데이터/인.허가/{path_json.parent.parent.stem}/{path_json.parent.stem}/{label['images'][0]['image.file.name']}"

    img = load_image_as_array(path_img)
    return img, df_label
