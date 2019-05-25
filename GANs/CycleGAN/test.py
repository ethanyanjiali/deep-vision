import tensorflow as tf

mse = tf.keras.losses.MeanAbsoluteError()

@tf.function
def train1():
    loss1 = mse(tf.random.normal([1, 3, 1]), tf.random.normal([1, 3, 1]))
    loss2 = mse(tf.random.normal([1, 3, 1]), tf.random.normal([1, 3, 1]))
    loss = loss1 + loss2
    print("inner", loss)
    return loss


def train_one_step():
    with tf.GradientTape() as tape:
        a = tf.random.normal([1, 3, 1])
        b = tf.random.normal([1, 3, 1])
        loss = mse(a, b)

    tf.print('inner tf print', loss)
    print("inner py print", loss)

    return loss


@tf.function
def train():
    loss = train_one_step()

    tf.print('outer tf print', loss)
    print('outer py print', loss)

    return loss

loss = train()
tf.print('outest tf print', loss)
print("outest py print", loss)
