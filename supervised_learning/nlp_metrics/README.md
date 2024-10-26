


![Project badge](/assets/pathway/003_color-0c5b38973bfe29fff8dd86f65a213ea2d2499a7d0d9e4549f440c50dc84c9618.png)100%# Natural Language Processing \- Evaluation Metrics

* Master
* By: Alexa Orrico, Software Engineer at Holberton School
* Weight: 1
* Your score will be updated as you progress.




* [Description](#description)
* [Quiz](#quiz)





[Go to tasks](#)

## Resources


**Read or watch:**


* [7 Applications of Deep Learning for Natural Language Processing](/rltoken/IYDIKPW-OMoU6MusXwtrIw "7 Applications of Deep Learning for Natural Language Processing")
* [10 Applications of Artificial Neural Networks in Natural Language Processing](/rltoken/2DOEX8NaohA50GuwdWtcGQ "10 Applications of Artificial Neural Networks in Natural Language Processing")
* [A Gentle Introduction to Calculating the BLEU Score for Text in Python](/rltoken/zsPawhP5ezqZDwFBc6DyYA "A Gentle Introduction to Calculating the BLEU Score for Text in Python")
* [Bleu Score](/rltoken/hPRNbFBuuRwZguNRoUpYlg "Bleu Score")
* [Evaluating Text Output in NLP: BLEU at your own risk](/rltoken/5Q8PN8uvw1fCVmjLdiEYbw "Evaluating Text Output in NLP: BLEU at your own risk")
* [ROUGE metric](/rltoken/7jWThnZo7x4BIaCzTYOzjg "ROUGE metric")
* [Evaluation and Perplexity](/rltoken/iWXkSDA5Cyw4bP_Z7VeFNg "Evaluation and Perplexity")
* [Evaluation metrics](/rltoken/fB5mQNcsuxScDvGo0S9deA "Evaluation metrics")


**Definitions to skim**


* [BLEU](/rltoken/5uK6B8SbX6SHp8t-H8NDFg "BLEU")
* [ROUGE](/rltoken/izJTokzam7qDlNjlIpDwJw "ROUGE")
* [Perplexity](/rltoken/gilvwXepRUooBDxu-3utrA "Perplexity")


**References:**


* [BLEU: a Method for Automatic Evaluation of Machine Translation (2002\)](/rltoken/lH3nYkepKlqhOFjI1h6Fmg "BLEU: a Method for Automatic Evaluation of Machine Translation (2002)")
* [ROUGE: A Package for Automatic Evaluation of Summaries (2004\)](/rltoken/hIWggM4NJ19XP-wAtik--g "ROUGE: A Package for Automatic Evaluation of Summaries (2004)")


## Learning Objectives


At the end of this project, you are expected to be able to [explain to anyone](/rltoken/JiW-QDEd5m2d5PzoZiH0rw "explain to anyone"), **without the help of Google**:


### General


* What are the applications of natural language processing?
* What is a BLEU score?
* What is a ROUGE score?
* What is perplexity?
* When should you use one evaluation metric over another?


## Requirements


### General


* Allowed editors: `vi`, `vim`, `emacs`
* All your files will be interpreted/compiled on Ubuntu 20\.04 LTS using `python3` (version 3\.9\)
* Your files will be executed with `numpy` (version 1\.25\.2\)
* All your files should end with a new line
* The first line of all your files should be exactly `#!/usr/bin/env python3`
* All of your files must be executable
* A `README.md` file, at the root of the folder of the project, is mandatory
* Your code should follow the `pycodestyle` style (version 2\.11\.1\)
* All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
* All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
* All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)
* You are not allowed to use the `nltk` module









**Great!**
 You've completed the quiz successfully! Keep going!
 (Hide quiz)




#### Question \#0



The BLEU score measures:



* A model’s accuracy
* A model’s precision
* A model’s recall
* A model’s perplexity







#### Question \#1



The ROUGE score measures:



* A model’s accuracy
* A model’s precision
* A model’s recall
* A model’s perplexity







#### Question \#2



Perplexity measures:



* The accuracy of a prediction
* The branching factor of a prediction
* A prediction’s recall
* A prediction’s accuracy







#### Question \#3



The BLEU score was designed for:



* Sentiment Analysis
* Machine Translation
* Question\-Answering
* Document Summarization







#### Question \#4



What are the shortcomings of the BLEU score?



* It cannot judge grammatical accuracy
* It cannot judge meaning
* It does not work with languages that lack word boundaries
* A higher score is not necessarily indicative of a better translation











## Tasks







### 0\. Unigram BLEU score




 mandatory
 











 Score: 100\.00% (Checks completed: 100\.00%)
 


Write the function `def uni_bleu(references, sentence):` that calculates the unigram BLEU score for a sentence:


* `references` is a list of reference translations
    + each reference translation is a list of the words in the translation
* `sentence` is a list containing the model proposed sentence
* Returns: the unigram BLEU score



```
$ cat 0-main.py
  #!/usr/bin/env python3
  
  uni_bleu = __import__('0-uni_bleu').uni_bleu
  
  references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
  sentence = ["there", "is", "a", "cat", "here"]
  
  print(uni_bleu(references, sentence))
  $ ./0-main.py
  0.6549846024623855
  $
  
```






**Repo:**


* GitHub repository: `holbertonschool-machine_learning`
* Directory: `supervised_learning/nlp_metrics`
* File: `0-uni_bleu.py`










 Help
 




×
#### Students who are done with "0\. Unigram BLEU score"

















 Review your work
 




×
#### Correction of "0\. Unigram BLEU score"







Start a new test
Close


















 Requirement success
 


 Requirement fail
 




 Code success
 


 Code fail
 




 Efficiency success
 


 Efficiency fail
 




 Text answer success
 


 Text answer fail
 




 Skipped \- Previous check failed
 







Get a sandbox

 QA Review
 




×
#### 0\. Unigram BLEU score












##### Commit used:


* **User:**  \-\-\-
* **URL:** Click here
* **ID:** `---`
* **Author:** \-\-\-
* **Subject:** *\-\-\-*
* **Date:** \-\-\-














**7/7** 
pts









### 1\. N\-gram BLEU score




 mandatory
 











 Score: 100\.00% (Checks completed: 100\.00%)
 


Write the function `def ngram_bleu(references, sentence, n):` that calculates the n\-gram BLEU score for a sentence:


* `references` is a list of reference translations
    + each reference translation is a list of the words in the translation
* `sentence` is a list containing the model proposed sentence
* `n` is the size of the n\-gram to use for evaluation
* Returns: the n\-gram BLEU score



```
$ cat 1-main.py
  #!/usr/bin/env python3
  
  ngram_bleu = __import__('1-ngram_bleu').ngram_bleu
  
  references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
  sentence = ["there", "is", "a", "cat", "here"]
  
  print(ngram_bleu(references, sentence, 2))
  $ ./1-main.py
  0.6140480648084865
  $
  
```






**Repo:**


* GitHub repository: `holbertonschool-machine_learning`
* Directory: `supervised_learning/nlp_metrics`
* File: `1-ngram_bleu.py`










 Help
 




×
#### Students who are done with "1\. N\-gram BLEU score"

















 Review your work
 




×
#### Correction of "1\. N\-gram BLEU score"







Start a new test
Close


















 Requirement success
 


 Requirement fail
 




 Code success
 


 Code fail
 




 Efficiency success
 


 Efficiency fail
 




 Text answer success
 


 Text answer fail
 




 Skipped \- Previous check failed
 







Get a sandbox

 QA Review
 




×
#### 1\. N\-gram BLEU score












##### Commit used:


* **User:**  \-\-\-
* **URL:** Click here
* **ID:** `---`
* **Author:** \-\-\-
* **Subject:** *\-\-\-*
* **Date:** \-\-\-














**7/7** 
pts









### 2\. Cumulative N\-gram BLEU score




 mandatory
 











 Score: 100\.00% (Checks completed: 100\.00%)
 


Write the function `def cumulative_bleu(references, sentence, n):` that calculates the cumulative n\-gram BLEU score for a sentence:


* `references` is a list of reference translations
    + each reference translation is a list of the words in the translation
* `sentence` is a list containing the model proposed sentence
* `n` is the size of the largest n\-gram to use for evaluation
* All n\-gram scores should be weighted evenly
* Returns: the cumulative n\-gram BLEU score



```
$ cat 2-main.py
  #!/usr/bin/env python3
  
  cumulative_bleu = __import__('2-cumulative_bleu').cumulative_bleu
  
  references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
  sentence = ["there", "is", "a", "cat", "here"]
  
  print(cumulative_bleu(references, sentence, 4))
  $ ./2-main.py
  0.5475182535069453
  $
  
```






**Repo:**


* GitHub repository: `holbertonschool-machine_learning`
* Directory: `supervised_learning/nlp_metrics`
* File: `2-cumulative_bleu.py`










 Help
 




×
#### Students who are done with "2\. Cumulative N\-gram BLEU score"

















 Review your work
 




×
#### Correction of "2\. Cumulative N\-gram BLEU score"







Start a new test
Close


















 Requirement success
 


 Requirement fail
 




 Code success
 


 Code fail
 




 Efficiency success
 


 Efficiency fail
 




 Text answer success
 


 Text answer fail
 




 Skipped \- Previous check failed
 







Get a sandbox

 QA Review
 




×
#### 2\. Cumulative N\-gram BLEU score












##### Commit used:


* **User:**  \-\-\-
* **URL:** Click here
* **ID:** `---`
* **Author:** \-\-\-
* **Subject:** *\-\-\-*
* **Date:** \-\-\-














**7/7** 
pts









[Previous project](/projects/2360)  



 Next project 







### Score

![Project badge](/assets/pathway/003_color-0c5b38973bfe29fff8dd86f65a213ea2d2499a7d0d9e4549f440c50dc84c9618.png)100%Congratulations! You made it!

Next project: Attention

[Open the next project](/projects/2338)


### Score

![Project badge](/assets/pathway/003_color-0c5b38973bfe29fff8dd86f65a213ea2d2499a7d0d9e4549f440c50dc84c9618.png)100%Congratulations! You made it!

Next project: Attention

[Open the next project](/projects/2338)



### Tasks list




* [Mandatory](#mandatory)
* [Advanced](#advanced)





0\. `Unigram BLEU score`
**100\.00%**


1\. `N-gram BLEU score`
**100\.00%**


2\. `Cumulative N-gram BLEU score`
**100\.00%**









×#### Recommended Sandbox

Running only### 1 image(0/5 Sandboxes spawned)

### Ubuntu 16\.04 \- Python 3\.5 \- Numpy

Ubuntu 16\.04 with Python 3\.5 and Numpy installed


Run
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





