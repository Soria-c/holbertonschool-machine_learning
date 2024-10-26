


![Project badge](/assets/pathway/003_color-0c5b38973bfe29fff8dd86f65a213ea2d2499a7d0d9e4549f440c50dc84c9618.png)0%# Time Series Forecasting

* Master
* By: Alexa Orrico, Software Engineer at Holberton School
* Weight: 4
* **Manual QA review must be done** (request it when you are done with the project)




* [Description](#description)





[Go to tasks](#)

![](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2020/7/3b16b59e54876f2cc4fe9dcf887ac40585057e2c.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20241026%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20241026T032810Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=f79296cdb2116933e7af340a652b5ab39ea7db56c53faf8788d0e6e1021934fa)


## Resources


**Read or watch:**


* [Time Series Prediction](/rltoken/IPHyUFv7WJ_cxTINQmtgNg "Time Series Prediction")
* [Time Series Forecasting](/rltoken/nIebldUW1xMYyP604lPmtA "Time Series Forecasting")
* [Time Series Talk : Stationarity](/rltoken/Z6gxuejq_ftwd64E-UPAYg "Time Series Talk : Stationarity")
* [tf.data: Build TensorFlow input pipelines](/rltoken/qwJQkXozMMU3FEi9upt-qw "tf.data: Build TensorFlow input pipelines")
* [Tensorflow Datasets](/rltoken/rQ78XqULPk8Ad6D__cnRWw "Tensorflow Datasets")


**Definitions to skim**


* [Time Series](/rltoken/nA9DP5YSKunSZOdceKp6pQ "Time Series")
* [Stationary Process](/rltoken/FcSWzi08147D5eQioh-lbw "Stationary Process")


**References:**


* [tf.keras.layers.SimpleRNN](/rltoken/0b-RuS_OStuKKP9vZspm3g "tf.keras.layers.SimpleRNN")
* [tf.keras.layers.GRU](/rltoken/SK2e7ZrNhZxC-vh-BeOhUQ "tf.keras.layers.GRU")
* [tf.keras.layers.LSTM](/rltoken/s6BR0PAxk1Ygn3eVTGm6pQ "tf.keras.layers.LSTM")
* [tf.data.Dataset](/rltoken/jRwxnbX7t2AvVDcnVJqg0w "tf.data.Dataset")


## Learning Objectives


At the end of this project, you are expected to be able to [explain to anyone](/rltoken/k12FbB4O3DKGTcHWiu6UdQ "explain to anyone"), **without the help of Google**:


### General


* What is time series forecasting?
* What is a stationary process?
* What is a sliding window?
* How to preprocess time series data
* How to create a data pipeline in tensorflow for time series data
* How to perform time series forecasting with RNNs in tensorflow


## Requirements


### General


* Allowed editors: `vi`, `vim`, `emacs`
* All your files will be interpreted/compiled on Ubuntu 20\.04 LTS using `python3` (version 3\.9\)
* Your files will be executed with `numpy` (version 1\.25\.2\), `tensorflow` (version 2\.15\) and pandas (version 2\.2\.2\)
* All your files should end with a new line
* The first line of all your files should be exactly `#!/usr/bin/env python3`
* All of your files must be executable
* A `README.md` file, at the root of the folder of the project, is mandatory
* Your code should follow the `pycodestyle` style (version 2\.11\.1\)
* All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
* All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
* All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)







## Tasks







### 0\. When to Invest




 mandatory
 






Bitcoin (BTC) became a trending topic after its [price](/rltoken/JGiW-D94g8VlAzwOBwdkuw "price") peaked in 2018\. Many have sought to predict its value in order to accrue wealth. Let’s attempt to use our knowledge of RNNs to attempt just that.


Given the [coinbase](/rltoken/vEVzC0M9D73iMNUZqf7Tpg "coinbase") and [bitstamp](/rltoken/JjaZZyvz3hChdFxPNbc3hA "bitstamp") datasets, write a script, `forecast_btc.py`, that creates, trains, and validates a keras model for the forecasting of BTC:


* Your model should use the past 24 hours of BTC data to predict the value of BTC at the close of the following hour (approximately how long the average transaction takes):
* The datasets are formatted such that every row represents a 60 second time window containing:
    + The start time of the time window in Unix time
    + The open price in USD at the start of the time window
    + The high price in USD within the time window
    + The low price in USD within the time window
    + The close price in USD at end of the time window
    + The amount of BTC transacted in the time window
    + The amount of Currency (USD) transacted in the time window
    + The [volume\-weighted average price](/rltoken/G0uAWRA037gh9RypZZBMqA "volume-weighted average price") in USD for the time window
* Your model should use an RNN architecture of your choosing
* Your model should use mean\-squared error (MSE) as its cost function
* You should use a `tf.data.Dataset` to feed data to your model


Because the dataset is [raw](/rltoken/bPiBmLbVH_WBtfKh0t8v5A "raw"), you will need to create a script, `preprocess_data.py` to preprocess this data. Here are some things to consider:


* Are all of the data points useful?
* Are all of the data features useful?
* Should you rescale the data?
* Is the current time window relevant?
* How should you save this preprocessed data?







**Repo:**


* GitHub repository: `holbertonschool-machine_learning`
* Directory: `supervised_learning/time_series`
* File: `README.md, forecast_btc.py, preprocess_data.py`










 Help
 




×
#### Students who are done with "0\. When to Invest"


















**0/3** 
pts









### 1\. Everyone wants to know




 mandatory
 






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







#### Add URLs here:






 Save
 















 Help
 




×
#### Students who are done with "1\. Everyone wants to know"


















**0/17** 
pts









[Previous project](/projects/3086)  



 Next project 







### Score

![Project badge](/assets/pathway/003_color-0c5b38973bfe29fff8dd86f65a213ea2d2499a7d0d9e4549f440c50dc84c9618.png)0%Please review **all the tasks** before you start the peer review.

Ready for my review


### Score

![Project badge](/assets/pathway/003_color-0c5b38973bfe29fff8dd86f65a213ea2d2499a7d0d9e4549f440c50dc84c9618.png)0%Please review **all the tasks** before you start the peer review.

Ready for my review



### Tasks list




* [Mandatory](#mandatory)
* [Advanced](#advanced)





0\. `When to Invest`



1\. `Everyone wants to know`










×#### Recommended Sandboxes

New sandbox * 
* US East (N. Virginia)
* Ubuntu 18\.04
* Ubuntu 22\.04
* 
* South America (São Paulo)
* Ubuntu 18\.04
* Ubuntu 22\.04
* 
* Europe (Paris)
* Ubuntu 18\.04
* Ubuntu 22\.04
* 
* Asia Pacific (Sydney)
* Ubuntu 18\.04
* Ubuntu 22\.04
No sandboxes yet!





