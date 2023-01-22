```python
from kaggle_datasets import KaggleDatasets

gcs_path = KaggleDatasets().get_gcs_path("tpu-getting-started")

filenames_tr = tf.io.gfile.glob(gcs_path + "/tfrecords-jpeg-512x512/train/*.tfrec") + tf.io.gfile.glob(gcs_path + "/tfrecords-jpeg-512x512/val/*.tfrec")
filenames_te = tf.io.gfile.glob(gcs_path + "/test/*.tfrec")
```