from pathlib import Path
import extcolors


dir = Path("/Users/jongbeom.kim/Downloads/New_sample/라벨링데이터")
for path_json in dir.glob("**/*.json"):
    img, df_label = get_image_and_label(path_json=path_json)
    extcolors.extract_from_path(path_img)
    df_pred = get_paddleocr_result(
        img=img,
        lang="ko",
        text_detection=True,
        text_recognition=True,
        converts_to_rects=True
    )
    
    f1_score = get_f1_score(df_label, df_pred, iou_thr=0.5)
    
    print(f"{path_json.stem} | f1 score: {f1_score}")

df_pred.head()

drawn = draw_rectangles_on_image(img=img, rectangles1=df_label, rectangles2=df_pred)
drawn2 = draw_rectangles_on_image(img=img, rectangles=df_pred)

show_image(drawn)
show_image(drawn2)


