
# Time Series Forecasting

## Resources


**Read or watch:**


* [Time Series Prediction](https://www.youtube.com/watch?v=d4Sn6ny_5LI "Time Series Prediction")
* [Time Series Forecasting](https://www.tensorflow.org/tutorials/structured_data/time_series "Time Series Forecasting")
* [Time Series Talk : Stationarity](https://www.youtube.com/watch?v=oY-j2Wof51c "Time Series Talk : Stationarity")
* [tf.data: Build TensorFlow input pipelines](https://www.tensorflow.org/guide/data "tf.data: Build TensorFlow input pipelines")
* [Tensorflow Datasets](https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/datasets.md "Tensorflow Datasets")


**Definitions to skim**


* [Time Series](https://en.wikipedia.org/wiki/Time_series "Time Series")
* [Stationary Process](https://en.wikipedia.org/wiki/Stationary_process "Stationary Process")


**References:**


* [tf.keras.layers.SimpleRNN](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/keras/layers/SimpleRNN.md "tf.keras.layers.SimpleRNN")
* [tf.keras.layers.GRU](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/keras/layers/GRU.md "tf.keras.layers.GRU")
* [tf.keras.layers.LSTM](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/keras/layers/LSTM.md "tf.keras.layers.LSTM")
* [tf.data.Dataset](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/data/Dataset.md "tf.data.Dataset")


## Learning Objectives


At the end of this project, you are expected to be able to [explain to anyone](/rltoken/k12FbB4O3DKGTcHWiu6UdQ "explain to anyone"), **without the help of Google**:


### General


* What is time series forecasting?
* What is a stationary process?
* What is a sliding window?
* How to preprocess time series data
* How to create a data pipeline in tensorflow for time series data
* How to perform time series forecasting with RNNs in tensorflow




## Tasks







### 0\. When to Invest



Bitcoin (BTC) became a trending topic after its [price](https://www.google.com/search?q=bitcoin+price "price") peaked in 2018\. Many have sought to predict its value in order to accrue wealth. Let’s attempt to use our knowledge of RNNs to attempt just that.


Given the [coinbase](./data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv "coinbase") and [bitstamp](./data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv"bitstamp") datasets, write a script, `forecast_btc.py`, that creates, trains, and validates a keras model for the forecasting of BTC:


* Your model should use the past 24 hours of BTC data to predict the value of BTC at the close of the following hour (approximately how long the average transaction takes):
* The datasets are formatted such that every row represents a 60 second time window containing:
    + The start time of the time window in Unix time
    + The open price in USD at the start of the time window
    + The high price in USD within the time window
    + The low price in USD within the time window
    + The close price in USD at end of the time window
    + The amount of BTC transacted in the time window
    + The amount of Currency (USD) transacted in the time window
    + The [volume\-weighted average price](https://en.wikipedia.org/wiki/Volume-weighted_average_price#:~:text=In%20finance%2C%20volume%2Dweighted%20average,traded%20over%20the%20trading%20horizon. "volume-weighted average price") in USD for the time window
* Your model should use an RNN architecture of your choosing
* Your model should use mean\-squared error (MSE) as its cost function
* You should use a `tf.data.Dataset` to feed data to your model


Because the dataset is [raw](https://en.wikipedia.org/wiki/Raw_data "raw"), you will need to create a script, `preprocess_data.py` to preprocess this data. Here are some things to consider:


* Are all of the data points useful?
* Are all of the data features useful?
* Should you rescale the data?
* Is the current time window relevant?
* How should you save this preprocessed data?






### 1\. Everyone wants to know





Everyone wants to know how to make money with BTC! Write a blog post explaining your process in completing the task above:


* An introduction to Time Series Forecasting
* An explanation of your preprocessing method and why you chose it
* An explanation of how you set up your `tf.data.Dataset` for your model inputs
* An explanation of the model architecture that you used
* A results section containing the model performance and corresponding graphs
* A conclusion of your experience, your thoughts on forecasting BTC, and a link to your github with the relevant code


Your posts should have examples and at least one picture, at the top. Publish your blog post on Medium or LinkedIn, and share it at least on LinkedIn.


When done, please add all URLs below (blog post, shared link, etc.)


Please, remember that these blogs **must be written in English** to further your technical ability in a variety of settings.

