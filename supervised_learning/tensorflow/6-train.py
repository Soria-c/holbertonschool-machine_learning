#!/usr/bin/env python3
"""Calculate Accuracy"""
import tensorflow.compat.v1 as tf


calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    builds, trains, and saves a neural network classifier
    ```
    Parameters:
    -----------
        X_train is a numpy.ndarray containing the training input data
        Y_train is a numpy.ndarray containing the training labels
        X_valid is a numpy.ndarray containing the validation input data
        Y_valid is a numpy.ndarray containing the validation labels
        layer_sizes is a list containing the number
            of nodes in each layer of the network
        activations is a list containing the activation
            functions for each layer of the network
        alpha is the learning rate
        iterations is the number of iterations to train over
        save_path designates where to save the model
    """
    x, y = create_placeholders(784, 10)
    y_pred = forward_prop(X_train, layer_sizes, activations)
    accuracy = calculate_accuracy(Y_train, y_pred)
    loss = calculate_loss(Y_train, y_pred)
    train_op = create_train_op(loss, alpha)
    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)
    tf.add_to_collection("y_pred", y_pred)
    tf.add_to_collection("loss", loss)
    tf.add_to_collection("accuracy", accuracy)
    tf.add_to_collection("train_op", train_op)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        log_info(loss, accuracy, X_train, Y_train,
                 X_valid, Y_valid, 0, sess, x, y)
        for i in range(iterations):
            sess.run(train_op, feed_dict={x: X_train, y: Y_train})
            if not ((i + 1) % 100):
                log_info(loss, accuracy, X_train, Y_train,
                         X_valid, Y_valid, i + 1, sess, x, y)
    return tf.train.Saver().save(sess, save_path)


def log_info(loss, accuracy, X_train, Y_train,
             X_valid, Y_valid, i, sess, x, y):
    """Logs training info"""
    print(f"After {i} iterations:")
    train_loss, train_accuracy = sess.run([loss, accuracy],
                                          feed_dict={x: X_train, y: Y_train})
    print(f"\tTraining Cost: {train_loss}")
    print(f"\tTraining Accuracy: {train_accuracy}")
    v_loss, v_accuracy = sess.run([loss, accuracy],
                                  feed_dict={x: X_valid, y: Y_valid})
    print(f"\tValidation Cost: {v_loss}")
    print(f"\tValidation Accuracy: {v_accuracy}")
