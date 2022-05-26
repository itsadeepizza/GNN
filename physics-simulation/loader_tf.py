"""
import tensorflow as tf


train_path = "dataset/water_drop/train.tfrecord"
test_path = "dataset/water_drop/test.tfrecord"
metadata_path =  "dataset/water_drop/metadata.json"

raw_dataset = tf.data.TFRecordDataset([test_path])
print(dir(raw_dataset))
print(raw_dataset[0])

for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)

"""


"""
import tfrecord
test_path = "dataset/water_drop/test.tfrecord"
loader = tfrecord.tfrecord_loader(test_path, None)
for record in loader:
    print(record)
"""

test_path = "dataset/water_drop/test.tfrecord"

import tensorflow as tf
import numpy as np
for record in tf.python_io.tf_record_iterator(test_path):
    example = tf.train.Example()
    example.ParseFromString(record)
    print(example)