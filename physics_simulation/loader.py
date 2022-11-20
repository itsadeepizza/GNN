# https://github.com/wu375/simple-physics-simulator-pytorch-geometry/blob/main/train_or_infer.py

import os
import json
import torch
from processor import Processor
from decoder import Decoder
from encoder import Encoder
from euler_integrator import integrator

def _read_metadata(data_path):
    metadata_path = os.environ['ROOT_DATASET'] + "/water_drop/metadata.json"
    with open(metadata_path, 'r') as f:
        return json.loads(f.read())


def prepare_data_from_tfds(data_path=os.environ['ROOT_DATASET'] + '/water_drop/train.tfrecord', is_rollout=False, batch_size=2, shuffle=True):
    import functools
    import tensorflow.compat.v1 as tf
    import tensorflow_datasets as tfds
    from lib import reading_utils
    import tree
    # from tfrecord.torch.dataset import TFRecordDataset
    def prepare_inputs(tensor_dict):
        pos = tensor_dict['position']
        pos = tf.transpose(pos, perm=[1, 0, 2])
        target_position = pos[:, -1]
        tensor_dict['position'] = pos[:, :-1]
        num_particles = tf.shape(pos)[0]
        tensor_dict['n_particles_per_example'] = num_particles[tf.newaxis]
        if 'step_context' in tensor_dict:
            tensor_dict['step_context'] = tensor_dict['step_context'][-2]
            tensor_dict['step_context'] = tensor_dict['step_context'][tf.newaxis]
        return tensor_dict, target_position

    def batch_concat(dataset, batch_size):
        windowed_ds = dataset.window(batch_size)
        initial_state = tree.map_structure(
            lambda spec: tf.zeros(shape=[0] + spec.shape.as_list()[1:], dtype=spec.dtype), dataset.element_spec)

        def reduce_window(initial_state, ds):
            return ds.reduce(initial_state, lambda x, y: tf.concat([x, y], axis=0))

        return windowed_ds.map(lambda *x: tree.map_structure(reduce_window, initial_state, x))

    def prepare_rollout_inputs(context, features):
        out_dict = {**context}
        pos = tf.transpose(features['position'], [1, 0, 2])
        target_position = pos[:, -1]
        out_dict['position'] = pos[:, :-1]
        out_dict['n_particles_per_example'] = [tf.shape(pos)[0]]
        if 'step_context' in features:
            out_dict['step_context'] = features['step_context']
        out_dict['is_trajectory'] = tf.constant([True], tf.bool)
        return out_dict, target_position

    metadata = _read_metadata('')
    ds = tf.data.TFRecordDataset([data_path])
    ds = ds.map(functools.partial(reading_utils.parse_serialized_simulation_example, metadata=metadata))
    if is_rollout:
        ds = ds.map(prepare_rollout_inputs)
    else:
        split_with_window = functools.partial(
            reading_utils.split_trajectory,
            window_length=6 + 1)
        ds = ds.flat_map(split_with_window)
        ds = ds.map(prepare_inputs)
        ds = ds.repeat()
        if shuffle:
            ds = ds.shuffle(512)
        ds = batch_concat(ds, batch_size)
    ds = tfds.as_numpy(ds)
    # for i in range(100):  # clear screen
    #     print()
    return ds


def prepare_data_from_tfds_test(data_path=os.environ['ROOT_DATASET'] + '/water_drop/train.tfrecord', is_rollout=False, batch_size=2, shuffle=True):
    # def prepare_data_from_tfds(data_path='data/train.tfrecord', is_rollout=False, batch_size=2):
        import functools
        import tensorflow.compat.v1 as tf
        import tensorflow_datasets as tfds
        from lib import reading_utils
        import tree
        from tfrecord.torch.dataset import TFRecordDataset
        def prepare_inputs(tensor_dict):
            pos = tensor_dict['position']
            pos = tf.transpose(pos, perm=[1, 0, 2])
            target_position = pos[:, -1]
            tensor_dict['position'] = pos[:, :-1]
            num_particles = tf.shape(pos)[0]
            tensor_dict['n_particles_per_example'] = num_particles[tf.newaxis]
            if 'step_context' in tensor_dict:
                tensor_dict['step_context'] = tensor_dict['step_context'][-2]
                tensor_dict['step_context'] = tensor_dict['step_context'][tf.newaxis]
            return tensor_dict, target_position

        def batch_concat(dataset, batch_size):
            windowed_ds = dataset.window(batch_size)
            initial_state = tree.map_structure(
                lambda spec: tf.zeros(shape=[0] + spec.shape.as_list()[1:], dtype=spec.dtype), dataset.element_spec)

            def reduce_window(initial_state, ds):
                return ds.reduce(initial_state, lambda x, y: tf.concat([x, y], axis=0))

            return windowed_ds.map(lambda *x: tree.map_structure(reduce_window, initial_state, x))

        def prepare_rollout_inputs(context, features):
            out_dict = { **context }
            pos = tf.transpose(features['position'], [1, 0, 2])
            target_position = pos[:, -1]
            out_dict['position'] = pos[:, :-1]
            out_dict['n_particles_per_example'] = [tf.shape(pos)[0]]
            if 'step_context' in features:
                out_dict['step_context'] = features['step_context']
            out_dict['is_trajectory'] = tf.constant([True], tf.bool)
            return out_dict, target_position

        metadata = _read_metadata('data/')
        ds = tf.data.TFRecordDataset([data_path])
        ds = ds.map(functools.partial(reading_utils.parse_serialized_simulation_example, metadata=metadata))
        if is_rollout:
            ds = ds.map(prepare_rollout_inputs)
        else:
            split_with_window = functools.partial(reading_utils.split_trajectory, window_length=6 + 1)
            ds = ds.flat_map(split_with_window)
            ds = ds.map(prepare_inputs)
            ds = ds.repeat()
            ds = ds.shuffle(512)
            ds = batch_concat(ds, batch_size)
        ds = tfds.as_numpy(ds)
        for i in range(100):  # clear screen
            print()
        return ds



if __name__ == "__main__":
    ds = prepare_data_from_tfds()
    device = torch.device("cpu")
    encoder = Encoder(device=device)
    proc = Processor(128, 128, 128, 128, M=10, device=device)
    decoder = Decoder().to(device)


    for features, labels in ds:
        features['position']                = torch.tensor(features['position']).to(device)
        features['n_particles_per_example'] = torch.tensor(features['n_particles_per_example']).to(device)
        features['particle_type']           = torch.tensor(features['particle_type']).to(device)
        labels                              = torch.tensor(labels).to(device)

        """
        n is the nuber of particles
        `partycle_type`: Integer values tensor of size n. Each value represent the material of the ith particle
        `position`: Float values tensor of size n x 6 x 2. It represents the last six positions (x, y) of the particles
        `n_particles_per_example`: Integer values Tensor of size 2 = [n1, n2] with n1 + n2 = n ????????? 

        `labels`: Float values tensor of size n x 2. It represents future positions to predict        
        """
        # import matplotlib.pyplot as plt
        if not all(features["particle_type"] == 5):
            print("AHIAhiAhi\n\n\n\n\n\nAhi")
        # for key, item in features.items():
        #     print(key)
        #     print(item.shape)
        position = features["position"]



        data = encoder(position)
        print(data)
        data = proc(data)
        print("Processed Data: ", data)
        acc = decoder(data)
        print("Acceleration:", acc)
        labels_est = integrator(position, acc)
        loss = torch.abs(labels_est - labels).sum()
        print("Loss is :", loss)


        # plt.scatter(x=position[:, 3, 0].numpy(), y=position[:, 3, 1].numpy(), color='y')
        # plt.scatter(x=position[:, 4, 0].numpy(), y=position[:, 4, 1].numpy(), color='g')
        # plt.scatter(x=position[:, 5, 0].numpy(), y=position[:, 5, 1].numpy(), color='r')
        # plt.scatter(x=labels[:, 0].numpy(), y=labels[:, 1].numpy(), color='b')
        # plt.show()
        input()
