#!/usr/bin/env python3
"""Learning Rate Decay"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None,
                early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent
    """
    def inverse_time_decay(epoch):
        """Inverse time decay"""
        return alpha / (1 + decay_rate * epoch)

    return network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=[
            K.callbacks.EarlyStopping(
                patience=patience
            ) if early_stopping else None,
            K.callbacks.LearningRateScheduler(
                schedule=inverse_time_decay,
                verbose=1
            ) if learning_rate_decay else None,
        ] if validation_data else None
    )
