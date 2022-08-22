# Mount Google Drive
```python
from google.colab import drive
import os
drive.mount("/content/drive")
os.chdir("/content")
# os.chdir("/content/drive/MyDrive/Libraries")
```

# Download Files to Local
```
from google.colab import files

files.download(path)
```

# Display Hangul
```python
import matplotlib as mpl
import matplotlib.pyplot as plt

%config InlineBackend.figure_format = "retina"
!apt -qq -y install fonts-nanum
fpath = "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"
fpath = "/NanumBarunGothic.ttf"
font = mpl.font_manager.FontProperties(fname=fpath, size=9)
plt.rc("font", family="NanumBarunGothic") 
mpl.font_manager._rebuild()
mpl.rcParams["axes.unicode_minus"] = False
```

# Prevent from Disconnecting
```
function ClickConnect(){
    console.log("코랩 연결 끊김 방지");
	document.querySelector("colab-toolbar-button#connect").click()}
setInterval(ClickConnect, 60*1000)
```

# Install Libraries Permanently
```python
!pip install --target=TARGET_PATH LIBRARY_NAME
```

# Use TPU in tensorflow
```python
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="grpc://" + os.environ["COLAB_TPU_ADDR"])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)
...
with strategy.scope():
	...
```
```python
strategy = tf.distribute.experimental.TPUStrategy(resolver)
with strategy.scope():
    model = create_model()
    hist = model.fit()
```

# Substitution for `cv2.imshow()`
```python
from google.colab.patches import cv2_imshow
```