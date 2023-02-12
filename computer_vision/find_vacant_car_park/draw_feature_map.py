from pathlib import Path
import scipy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras import Model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model


def build_model_for_feature_map():
    model_pretrained = load_model("model.h5")

    model_feature_map = Model(
        inputs=model_pretrained.input,
        outputs=(
            model_pretrained.layers[-3].output,
            model_pretrained.layers[-1].output
        )
    )
    return model_feature_map


def draw_output(img, final_output, pred):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.imshow(img)
    # ax.imshow(final_output, cmap="jet", alpha=0.2)
    ax.imshow(final_output, cmap="gray", alpha=0.8)
    ax.set_title(f"Prediction: {pred[0][1]:.2%} Free")
    ax.axis("off")

    plt.show()


def draw_image_with_feature_map(img, model_feature_map):
    test_img = Image.open(img)
    test_img = test_img.resize((224, 224))

    test_input = img_to_array(test_img)
    test_input = np.expand_dims(test_input, axis=0)
    test_input = preprocess_input(test_input)

    last_conv_output, pred = model_feature_map.predict(test_input)

    # (1, 7, 7, 1280) -> (7, 7, 1280)
    last_conv_output = np.squeeze(last_conv_output)
    # (7, 7, 1280) -> (224, 224, 1280)
    feature_activation_maps = scipy.ndimage.zoom(
        input=last_conv_output, zoom=(32, 32, 1), order=1
    )

    weight_last = model_feature_map.layers[-1].get_weights()[0] # (1280, 2)
    # label_predicted = np.argmax(pred)
    # weight_free = weight_last[:, label_predicted] # (1280, 1)
    weight_free = weight_last[:, 1] # (1280, 1)

    # (224 * 224, 1280) * (1280, 1) -> (224 * 224, 1)
    final_output = np.dot(
        feature_activation_maps.reshape((224 * 224, 1280)),
        weight_free
    ).reshape((224, 224))

    draw_output(test_img, final_output, pred)


def main():
    model_feature_map = build_model_for_feature_map()

    draw_image_with_feature_map(
        model_feature_map=model_feature_map,
        # img="/Users/jongbeom.kim/Downloads/data/Free/img_921202603.jpg"
        img="/Users/jongbeom.kim/Downloads/data/Full/img_1002012601.jpg"
    )


if __name__ == "__main__":
    main()
