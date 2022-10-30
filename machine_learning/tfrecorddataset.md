- Reference: https://www.kaggle.com/code/cdeotte/cutmix-and-mixup-on-gpu-tpu
# Create `TFRecordDataset`
```python
AUTO = tf.data.experimental.AUTOTUNE

ds = tf.data.TFRecordDataset(filenames_tr, num_parallel_reads=AUTO)
```

# Ordering Option
```python
ignore_order = tf.data.Options()
# Increase speed by disabling ordering.
ignore_order.experimental_deterministic = False
ds = ds.with_options(ignore_order)
```

# `tf.io.FixedLenFeature()`
```python
def read_labeled_tfrecord(example):
    labeled_tfrec_format = {
        # `list()` means single element.
        "image": tf.io.FixedLenFeature(list(), tf.string),
        "class": tf.io.FixedLenFeature(list(), tf.int64)
    }
    example = tf.io.parse_single_example(example, labeled_tfrec_format)
    img = decode_image(example["image"])
    label = tf.cast(example["class"], dtype=tf.int32)
    return img, label


ds = ds.map(read_labeled_tfrecord, num_parallel_calls=AUTO)
```

# Iterate `TFRecordDataset`
```python
# Example
for idx, (img, label) in enumerate(ds):
    ...
```