import tensorflow.compat.v1 as tf
import numpy as np
import functools
import json
from deepmind import reading_utils

INPUT_SEQUENCE_LENGTH = 6  # So we can calculate the last 5 velocities.
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3

test_path = "dataset/water_drop/test.tfrecord"
metadata_path =  "dataset/water_drop/metadata.json"

# ds = tf.data.TFRecordDataset([test_path])
# with open(metadata_path, "r") as f:
#     metadata = json.loads(f.read())
# ds = ds.map(functools.partial(
#     reading_utils.parse_serialized_simulation_example, metadata=metadata))
# split_with_window = functools.partial(
#       reading_utils.split_trajectory,
#       window_length=INPUT_SEQUENCE_LENGTH + 1)
# ds = ds.flat_map(split_with_window)
# # Splits a chunk into input steps and target steps
# print(ds)

from deepmind.train import get_input_fn
f = get_input_fn("dataset/water_drop", 1, 'one_step_train', 'test')
ds = f()
print(list(ds.take(1).as_numpy_iterator()))
