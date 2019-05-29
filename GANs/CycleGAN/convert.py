"""
This module converts a TF 2 model to a TF Lite file
"""
import tensorflow as tf


def convert():
    saved_mode_dir = './saved_models/celeba/male2female/201905282111/'
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_mode_dir)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_model = converter.convert()
    open("./converted_model.tflite", "wb").write(tflite_model)
    return


if __name__ == '__main__':
    convert()