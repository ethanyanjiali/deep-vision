import tensorflow as tf

from models.mobilenet_v1 import MobileNetV1

print(tf.__version__)

training_configs = {
    'mobilenetv1_1.0': {
        'batch_size_per_replica': 32,
        'model': MobileNetV1,
        'alpha': 1.0,
        'epochs': 10,
    }
}


def main():
    strategy = tf.distribute.MirroredStrategy()
    num_replicas = strategy.num_replicas_in_sync
    print('Number of devices: {}'.format(num_replicas))

    config = training_configs['mobilenetv1_1.0']
    batch_size_per_replica = config['batch_size_per_replica']
    global_batch_size = batch_size_per_replica * num_replicas
    Model = config['model']
    epochs = config['epochs']

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    with strategy.scope():
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE)
        model = Model(input_shape=(224, 224, 3))
        # Trainable params: 4,242,856
        model.summary()

        def compute_loss(labels, predictions):
            per_example_loss = loss_object(labels, predictions)
            return tf.nn.compute_average_loss(
                per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

        def train_step(inputs):
            images, labels = inputs

            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = compute_loss(labels, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))

            return loss

        def validate_step(inputs):
            images, labels = inputs

            predictions = model(images, training=False)
            loss = compute_loss(labels, predictions)
            return loss

        @tf.function
        def distributed_train_step(dataset_inputs):
            per_replica_losses = strategy.experimental_run_v2(
                train_step, args=(dataset_inputs, ))
            return strategy.reduce(
                tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


if __name__ == "__main__":
    main()