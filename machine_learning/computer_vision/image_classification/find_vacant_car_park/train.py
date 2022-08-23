from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint


def prepare_dataset(dir):
    gen_tr = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8, 1.2],
        shear_range=0.01,
        zoom_range=[0.9, 1.1],
        validation_split=0.1,
        preprocessing_function=preprocess_input
    )
    gen_val = ImageDataGenerator(
        validation_split=0.1,
        preprocessing_function=preprocess_input
    )

    ds_tr = gen_tr.flow_from_directory(
        dir,
        target_size=(224,224),
        classes=["Full","Free"],
        class_mode="categorical",
        batch_size=32,
        shuffle=True,
        subset="training"
    )
    ds_val = gen_val.flow_from_directory(
        dir,
        target_size=(224,224),
        classes=["Full","Free"],
        class_mode="categorical",
        batch_size=32,
        shuffle=False,
        subset="validation"
    )
    return ds_tr, ds_val


def build_model():
    model_mobilenet = MobileNetV2(
        input_shape=(224, 224, 3),
        weights="imagenet",
        include_top=False
    )
    z = model_mobilenet.output
    z = GlobalAveragePooling2D()(z)
    outputs = Dense(units=2, activation="softmax")(z)

    model = Model(inputs=model_mobilenet.input, outputs=outputs)
    for layer in model.layers:
        layer.trainable = True

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["acc"]
    )

    model.summary()
    return model


def train(model, training_dataset, validation_dataset):
    mc = ModelCheckpoint(
        filepath="model.h5",
        monitor="val_acc",
        save_best_only=True,
        verbose=1
    )

    history = model.fit_generator(
        training_dataset,
        validation_data=validation_dataset,
        epochs=10,
        callbacks=[es, mc]
    )


def main():
    ds_tr, ds_val = prepare_dataset(dir)
    model = build_model()


if __name__ == "__main__":
    main()
