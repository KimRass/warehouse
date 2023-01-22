# CIFAR-10
```python
(X_tr, y_tr), (X_te, y_te) = tf.keras.datasets.cifar10.load_data()
y_tr_val = to_categorical(y_tr_val)
y_te = to_categorical(y_te)

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
```
# Fashion MNIST
```python
# TensorFlow
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# PyTorch
torchvision.datasets.MNIST(root="...", train=True, download=True, transform=...)
```
# COCO (Common Objects in COntext)
- 83 classes: `"aeroplane"`, `"apple"`, `"backpack"`, `"banana"`, `"baseball bat"`, `"baseball glove"`, `"bear"`, `"bed"`, `"bench"`, `"bicycle"`, `"bird"`, `"boat"`, `"book"`, `"bottle"`, `"bowl"`, `"broccoli"`, `"bus"`, `"cake"`, `"car"`, `"carrot"`, `"cat"`, `"cell phone"`, `"chair"`, `"clock"`, `"cow"`, `"cup"`, `"diningtable"`, `"dog"`, `"dog"`, `"donut"`, `"elephant"`, `"fire hydrant"`, `"fork"`, `"frisbee"`, `"giraffe"`, `"glass"`, `"hair drier"`, `"handbag"`, `"horse"`, `"hot"`, `"keyboard"`, `"kite"`, `"knife"`, `"laptop"`, `"microwave"`, `"motorbike"`, `"mouse"`, `"orange"`, `"oven"`, `"parking meter"`, `"person"`, `"pizza"`, `"pottedplant"`, `"refrigerator"`, `"remote"`, `"sandwich"`, `"scissors"`, `"sheep"`, `"sign"`, `"sink"`, `"skateboard"`, `"skis"`, `"snowboard"`, `"sofa"`, `"spoon"`, `"sports ball"`, `"stop"`, `"suitcase"`, `"surfboard"`, `"teddy bear"`, `"tennis racket"`, `"tie"`, `"toaster"`, `"toilet"`, `"toothbrush"`, `"traffic light"`, `"train"`, `"truck"`, `"tvmonitor"`, `"umbrella"`, `"vase"`, `"wine"`, `"zebra"`