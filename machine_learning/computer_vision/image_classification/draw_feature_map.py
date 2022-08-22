from distutils.command.build import build
import os
from pathlib import Path
import scipy
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint


def build_model_for_feature_map():
    model_pretrained = load_model("model.h5")

    model = Model(
        inputs=model_pretrained.input,
        outputs=(
            model_pretrained.layers[-3].output,
            model_pretrained.layers[-1].output
        )
    )
    # model.summary()
    return model


def draw_image_with_feature_map(img, model_feature_map):
    test_img = img_to_array(
        load_img(img), target_size=(224, 224)
    )
    test_input = preprocess_input(np.expand_dims(test_img.copy(), axis=0))
    # pred = model.predict(test_input)

    last_conv_output, pred = model_feature_map.predict(test_input)

    last_conv_output = np.squeeze(last_conv_output) # (7,7, 1280)
    # (7,7,1280) -> (224,224,1280)
    feature_activation_maps = scipy.ndimage.zoom(last_conv_output, (32, 32, 1), order=1) 

    pred_class = np.argmax(pred) # 0: Full, 1:Free
    last_weight = model_feature_map.layers[-1].get_weights()[0] # (1280, 2)
    predicted_class_weights = last_weight[:, pred_class] # (1280, 1)

    final_output = np.dot(
        feature_activation_maps.reshape((224 * 224, 1280)),
        predicted_class_weights
    ).reshape(224, 224)
    # (224*224, 1280) dot_product(1280, 1) = (224*224, 1)
    
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(16, 20)

    ax[0].imshow(test_img.astype(np.uint8))
    ax[0].set_title(parking.split("\\")[1])
    ax[0].axis("off")

    ax[1].imshow(test_img.astype(np.uint8), alpha=0.5)
    ax[1].imshow(final_output, cmap="jet", alpha=0.5)
    ax[1].set_title("%.2f%% Free, class activation map" % (pred[0][1]*100))
    ax[1].axis("off")
    plt.show()


def main():
    model_feature_map = build_model_for_feature_map()

    draw_image_with_feature_map(
        model_feature_map=model_feature_map,
        img="/Users/jongbeom.kim/Downloads/data/Free/img_928192103.jpg",
    )


if __name__ == "__main__":
    main()
