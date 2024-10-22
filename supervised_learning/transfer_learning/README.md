# Transfer Learning
## Resources
* [A Comprehensive Hands-on Guide to Transfer Learning with Real-World Applications in Deep Learning](https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a)
* [Transfer Learning](https://www.youtube.com/watch?v=FQM13HkEfBk&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=21)
* [Transfer learning & fine-tuning](https://www.tensorflow.org/guide/keras/transfer_learning/)
* [Transfer learning](https://en.wikipedia.org/wiki/Transfer_learning)
* [Keras Applications](https://keras.io/api/applications/)
* [Keras Datasets](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/)
* [tf.keras.layers.Lambda](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Lambda)
* [tf.image.resize](https://www.tensorflow.org/api_docs/python/tf/image/resize)
* [A Survey on Deep Transfer Learning](https://arxiv.org/pdf/2403.17561)
## Learning Objectives
* What is a transfer learning?
* What is fine-tuning?
* What is a frozen layer? How and why do you freeze a layer?
* How to use transfer learning with Keras applications
## 0. Transfer Knowledge
Write a python script that trains a convolutional neural network to classify the CIFAR 10 dataset:

Keras pakages a number of deep leanring models alongside pre-trained weights into an applications module.

* You must use one of the applications listed in [Keras Applications](https://keras.io/api/applications/)
* Your script must save your trained model in the current working directory as cifar10.h5
* Your saved model should be compiled
* Your saved model should have a validation accuracy of 87% or higher
* Your script should not run when the file is imported
* **Hint1**: The training and tweaking of hyperparameters may take a while so start early!
* **Hint2**: The CIFAR 10 dataset contains 32x32 pixel images, however most of the Keras applications are trained on much larger images. Your first layer should be a lambda layer that scales up the data to the correct size
* **Hint3**: You will want to freeze most of the application layers. Since these layers will always produce the same output, you should compute the output of the frozen layers ONCE and use those values as input to train the remaining trainable layers. This will save you A LOT of time.

In the same file, write a function `def preprocess_data(X, Y):` that pre-processes the data for your model:

* `X` is a `numpy.ndarray` of shape `(m, 32, 32, 3)` containing the CIFAR 10 data, where m is the number of data points
* `Y` is a `numpy.ndarray` of shape `(m,)` containing the CIFAR 10 labels for `X`
* Returns: `X_p`, `Y_p`
    * `X_p` is a `numpy.ndarray` containing the preprocessed X
    * `Y_p` is a `numpy.ndarray` containing the preprocessed Y

```python
#!/usr/bin/env python3

from tensorflow import keras as K
preprocess_data = __import__('0-transfer').preprocess_data

# to fix issue with saving keras applications
K.learning_phase = K.backend.learning_phase 

_, (X, Y) = K.datasets.cifar10.load_data()
X_p, Y_p = preprocess_data(X, Y)
model = K.models.load_model('cifar10.h5')
model.evaluate(X_p, Y_p, batch_size=128, verbose=1)
```
```txt
10000/10000 [==============================] - 159s 16ms/sample - loss: 0.3329 - acc: 0.8864
```
## 1. "Research is what I'm doing when I don't know what I'm doing." - Wernher von Braun
Write a blog post explaining your experimental process in completing the task above written as a journal-style scientific paper:
| Experimental process         | Section of Paper           |
|------------------------------|----------------------------|
| What did I do in a nutshell? | Abstract                   |
| What is the problem?         | Introduction               |
| How did I solve the problem? | Materials and Methods      |
| What did I find out?         | Results                    |
| What does it mean?           | Discussion                 |
| Who helped me out?           | Acknowledgments (optional) |
| Whose work did I refer to?   | Literature Cited           |
| Extra Information            | Appendices (optional)      |

Your posts should have examples and at least one picture, at the top. Publish your blog post on Medium or LinkedIn, and share it at least on LinkedIn.

When done, please add all URLs below (blog post, tweet, etc.)

Please, remember that these blogs must be written in English to further your technical ability in a variety of settings.