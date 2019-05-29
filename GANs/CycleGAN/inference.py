import glob
import os

from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

from models import make_generator_model, make_discriminator_model


def main():
    generator_a2b = make_generator_model(n_blocks=9)
    generator_b2a = make_generator_model(n_blocks=9)

    checkpoint_dir = './checkpoints-celeba'
    checkpoint = tf.train.Checkpoint(generator_a2b=generator_a2b,
                                     generator_b2a=generator_b2a)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    # Load TFLite model and allocate tensors.
    # interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
    # interpreter.allocate_tensors()

    # Get input and output tensors.
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()

    def generate(filename, idx, kind):
        with open(filename, 'rb') as f:
            content = f.read()
        original = tf.image.decode_jpeg(content)
        resized_original = tf.image.resize(original, (256, 256))
        float_original = tf.cast(resized_original, tf.float32)
        inputs = float_original / 127.5 - 1
        inputs = tf.expand_dims(inputs, 0)

        # outputs = generator_b2a(inputs)
        outputs = generator_a2b(inputs)
        # interpreter.set_tensor(input_details[0]['index'], inputs)
        # interpreter.invoke()
        # outputs = interpreter.get_tensor(output_details[0]['index'])

        generated = outputs[0]
        generated = (generated + 1) * 127.5
        generated = tf.cast(generated, tf.uint8)
        # plt.imshow(original)
        # plt.show()
        # plt.imshow(generated)
        # plt.show()
        # print(original)
        im1 = Image.fromarray(original.numpy())
        im1.save('samples_celeba/{}_{}_original.JPEG'.format(kind, idx))
        generated = tf.image.resize(generated, (218, 178))
        generated = tf.cast(generated, tf.uint8)
        im2 = Image.fromarray(generated.numpy())
        im2.save('samples_celeba/{}_{}_generated.JPEG'.format(kind, idx))

    # files = glob.glob(os.path.join('datasets', 'horse2zebra', 'testB', '*'))
    generate('/Users/yanjia.li/Downloads/img_align_celeba/000030.jpg', 30, 'male2female')
    # for idx in range(120):
    #    generate(files[idx], idx, 'male2female')

    # tf.saved_model.save(generator_a2b, "./saved_models/celeba/male2female/201905282111/")

    # tf.saved_model.save(generator_b2a, "./saved_models/celeba/female2male/201905282111/")


if __name__ == '__main__':
    main()

