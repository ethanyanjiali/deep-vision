import random
import tensorflow as tf


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
                    tmp = tf.identity(self.pool[random_id])
                    self.pool[random_id] = image
                    return_images.append(tmp)
                else:
                    # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return return_images
