#!/usr/bin/env python3
"""Early Stopping"""


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None,
                early_stopping=False, patience=0, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent
    """
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
            )
        ] if early_stopping and validation_data else None
    )
