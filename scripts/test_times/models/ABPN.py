
''' abpn '''
import os
import sys
import argparse
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import backend as K
try:
    from scripts.test_times.convert import convert_model
except Exception:
    sys.path.append(sys.path[0].replace('scripts/test_times/models', ''))
    from scripts.test_times.convert import convert_model


def abpn(num_block: int, num_channel: int, scale: int):
    ''' abpn model '''
    inp = layers.Input(shape=(None, None, 3))
    upsample_func = layers.Lambda(lambda x_list: tf.concat(x_list, axis=3))     # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
    upsampled_inp = upsample_func([inp]*(scale**2))

    # Feature extraction
    x = layers.Conv2D(num_channel, 3, padding='same', activation='relu')(inp)

    for _ in range(num_block):
        x = layers.Conv2D(num_channel, 3, padding='same', activation='relu')(x)

    # Pixel-Shuffle
    x = layers.Conv2D(3*(scale**2), 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(3*(scale**2), 3, padding='same')(x)
    x = layers.Add()([upsampled_inp, x])

    out = layers.Lambda(lambda x: tf.nn.depth_to_space(x, scale))(x)
    out = layers.Lambda(lambda x: K.clip(x, 0., 255.))(out)

    return models.Model(inputs=inp, outputs=out)


def main():
    ''' main '''
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--blocks', required=True, type=int, help='blocks')
    parser.add_argument('--channels', required=True, type=int, help='channels')
    parser.add_argument('--scale', required=True, type=int, help='scale')
    args = parser.parse_args()

    # build model
    model = abpn(args.blocks, args.channels, args.scale)
    model.summary()

    # prepare
    name = f'M{args.blocks}C{args.channels}_X{args.scale}'
    os.makedirs('experiments/test_times', exist_ok=True)
    os.makedirs('experiments/test_times/models', exist_ok=True)
    os.makedirs('experiments/test_times/models/ABPN', exist_ok=True)
    os.makedirs('experiments/test_times/models/ABPN/int8', exist_ok=True)

    # to tflite
    model.save(f'/tmp/ABPN/{name}', overwrite=True, include_optimizer=True, save_format='tf')
    convert_model(f'/tmp/ABPN/{name}', f'experiments/test_times/models/ABPN/{name}.tflite', args.scale, time=True)

    # to int8 tflite
    convert_model(f'/tmp/ABPN/{name}', f'experiments/test_times/models/ABPN/int8/{name}.tflite', args.scale, mode='int8', time=True)


if __name__ == '__main__':
    main()
