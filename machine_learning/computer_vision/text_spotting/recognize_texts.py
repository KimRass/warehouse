from paddleocr import PaddleOCR
import pandas as pd
import numpy as np


def convert_quadrilaterals_to_rectangles(df):
    df.insert(1, "ymax", df[["y1", "y2", "y3", "y4"]].max(axis=1))
    df.insert(1, "xmax", df[["x1", "x2", "x3", "x4"]].max(axis=1))
    df.insert(1, "ymin", df[["y1", "y2", "y3", "y4"]].min(axis=1))
    df.insert(1, "xmin", df[["x1", "x2", "x3", "x4"]].min(axis=1))
    df.drop(["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"], axis=1, inplace=True)

    df[["xmin", "ymin", "xmax", "ymax"]] = df[["xmin", "ymin", "xmax", "ymax"]].astype("int")
    return df


def get_paddleocr_result(
    img,
    lang="en",
    text_detection=True,
    text_recognition=True,
    converts_to_rects=False
):
    if lang == "ko":
        lang = "korean"
    elif lang == "ja":
        lang = "japan"

    ocr = PaddleOCR(lang=lang)
    result = ocr.ocr(img=img, det=text_detection, rec=text_recognition, cls=False)

    if text_detection and not text_recognition:
        df_result = pd.DataFrame(
            np.array(result).reshape(-1, 8),
            columns=["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"],
            dtype="int"
        )
    elif not text_detection and text_recognition:
        return pd.DataFrame(
            result,
            columns=["text", "confidence"]
        )
    elif text_detection and text_recognition:
        cols = ["text", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
        if result:
            df_result = pd.DataFrame(
                [(row[1][0], *list(map(int, sum(row[0], [])))) for row in result],
                columns=cols
            )
        else:
            df_result = pd.DataFrame(columns=cols)

    if converts_to_rects:
        df_result = convert_quadrilaterals_to_rectangles(df_result)
    return df_result
