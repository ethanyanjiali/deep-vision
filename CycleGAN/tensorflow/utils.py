import random
import tensorflow as tf


class LinearDecay(tf.optimizers.schedules.LearningRateSchedule):
    # if `step` < `step_decay`: use fixed learning rate
    # else: linearly decay the learning rate to zero
    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step >= self._step_decay,
            true_fn=lambda: self._initial_learning_rate * (1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)),
            false_fn=lambda: self._initial_learning_rate
        ))
        return self.current_learning_rate

    def get_config(self):
        return {
            'initial_learning_rate': self._initial_learning_rate,
            'total_steps': self._steps,
            'step_decay': self._step_decay,
        }


# This image pool only works in TF eager mode. Not graph mode (tf.function)
class ImagePool:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.count = 0
        self.pool = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            # if the buffer is not full; keep inserting current images to the buffer
            if self.count < self.pool_size:
                self.count = self.count + 1
                self.pool.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    # by 50% chance, the buffer will return a previously stored image
                    # and insert the current image into the buffer
                    # randint is inclusive
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.pool[random_id]
                    self.pool[random_id] = image
                    return_images.append(tmp)
                else:
                    # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return tf.stack(return_images, axis=0)
