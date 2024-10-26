


![Project badge](/assets/pathway/003_color-0c5b38973bfe29fff8dd86f65a213ea2d2499a7d0d9e4549f440c50dc84c9618.png)0%# QA Bot

* Master
* By: Alexa Orrico, Software Engineer at Holberton School
* Weight: 1
* **Manual QA review must be done** (request it when you are done with the project)




* [Description](#description)





[Go to tasks](#)

![](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2020/11/52f1c0b37c7c58d30706b7e24bb7534b5a5889db.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20241026%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20241026T033816Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=fcd2a72e41eb01bb3802085def463edd97fda91208454529023a845cab048d65)


## Resources


**Read or watch**: 


* [Question Answering (Q\&A) Systems with LLMs](/rltoken/aOj6Tsmjt9GcLI9UBcfwSQ "Question Answering (Q&A) Systems with LLMs")
* [Improving Language Understanding by Generative Pre\-Training (2018\)](/rltoken/vFBWBmL-idNfApsMEt-RRw "Improving Language Understanding by Generative Pre-Training (2018)")
* [BERT: Pre\-training of Deep Bidirectional Transformers for Language Understanding (2018\)](/rltoken/QpwiBQGAz5Juui5_JurXhw "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018)")
* [SQuAD 2\.0](/rltoken/vigSZvwAnpAoqGUHBQ6TIw "SQuAD 2.0")
* [Know What You Don’t Know: Unanswerable Questions for SQuAD (2018\)](/rltoken/pVvNxpeU4kWEfwsj5OadQA "Know What You Don’t Know: Unanswerable Questions for SQuAD (2018)")
* [GLUE Benchmark](/rltoken/KDjvnz6EKqTiRjQwhQ47tQ "GLUE Benchmark")
* [GLUE: A Multi\-Task Benchmark and Analysis Platform for Natural Language Understanding (2019\)](/rltoken/zH4c1olq3tnR3DFd-ZA8fQ "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding (2019)")
* [Speech\-transformer: A no\-recurrence sequence\-to\-sequence model for speech recognition (2018\)](/rltoken/3Izjidm3dRwMCq1qdD3Gzw "Speech-transformer: A no-recurrence sequence-to-sequence model for speech recognition (2018)")


**More recent papers in NLP:**


* [Generating Long Sequences with Sparse Transformers (2019\)](/rltoken/cG4JFWqDglifBDx5j4f_Fw "Generating Long Sequences with Sparse Transformers (2019)")
* [Transformer\-XL: Attentive Language Models Beyond a Fixed\-Length Context (2019\)](/rltoken/6FpvA7gO2igFbD3pae_adQ "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context (2019)")
* [XLNet: Generalized Autoregressive Pretraining for Language Understanding (2019\)](/rltoken/VToc2xjHOOrqoiE8kOqDjQ "XLNet: Generalized Autoregressive Pretraining for Language Understanding (2019)")
* [Language Models are Unsupervised Multitask Learners (GPT\-2, 2019\)](/rltoken/E6AzWVl0M01VKZGmNbQsDw "Language Models are Unsupervised Multitask Learners (GPT-2, 2019)")
* [Language Models are Few\-Shot Learners (GPT\-3, 2020\)](/rltoken/NElHlTRbjzoZTYMBIF_Ayw "Language Models are Few-Shot Learners (GPT-3, 2020)")
* [ALBERT: A Lite BERT for Self\-Supervised Learning of Language Representations (2020\)](/rltoken/ADxo0AT5MUV1Yqq_Gmf4Uw "ALBERT: A Lite BERT for Self-Supervised Learning of Language Representations (2020)")


To keep up with the newest papers and their code bases go to [paperswithcode.com](/rltoken/wtaWMfzETYugkLtPfcCq1g "paperswithcode.com"). For example, check out the [raked list of state of the art models for Language Modelling on Penn Treebank](/rltoken/nHZBAo1Nhdo63LdNsr0hGA "raked list of state of the art models for Language Modelling on Penn Treebank").


## Learning Objectives


At the end of this project, you are expected to be able to [explain to anyone](/rltoken/WzIBN64E6sbdFRApZ4_hLg "explain to anyone"), **without the help of Google**:


### General


* What is Question\-Answering?
* What is Semantic Search?
* What is BERT?
* How to develop a QA chatbot
* How to use the `transformers` library
* How to use the `tensorflow-hub` library


## Requirements


### General


* Allowed editors: `vi`, `vim`, `emacs`
* All your files will be interpreted/compiled on Ubuntu 20\.04 LTS using `python3` (version 3\.9\)
* Your files will be executed with `numpy` (version 1\.25\.2\) and `tensorflow` (version 2\.15\)
* All your files should end with a new line
* The first line of all your files should be exactly `#!/usr/bin/env python3`
* All of your files must be executable
* A `README.md` file, at the root of the folder of the project, is mandatory
* Your code should follow the `pycodestyle` style (version 2\.11\.1\)
* All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
* All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
* All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)


## Install Tensorflow Hub



```
pip install --user tensorflow-hub==0.15.0
  
```

## Install Transformers



```
pip install --user transformers==4.44.2
  
```

## Zendesk Articles


For this project, we will be using a collection of Holberton USA Zendesk Articles, [ZendeskArticles.zip](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/misc/2020/11/c15a067b44a328c7d5a03c79070b7865f444d1e3.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20241026%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20241026T033816Z&X-Amz-Expires=345600&X-Amz-SignedHeaders=host&X-Amz-Signature=c1627a128d1f43e77cee3818bcf9ab357d43c5e7e926307291a9f93497adfdaa "ZendeskArticles.zip").







## Tasks







### 0\. Question Answering




 mandatory
 






Write a function `def question_answer(question, reference):` that finds a snippet of text within a reference document to answer a question:


* `question` is a string containing the question to answer
* `reference` is a string containing the reference document from which to find the answer
* Returns: a string containing the answer
* If no answer is found, return `None`
* Your function should use the `bert-uncased-tf2-qa` model from the `tensorflow-hub` library
* Your function should use the pre\-trained `BertTokenizer`, `bert-large-uncased-whole-word-masking-finetuned-squad`, from the `transformers` library



```
$ cat 0-main.py
  #!/usr/bin/env python3
  
  question_answer = __import__('0-qa').question_answer
  
  with open('ZendeskArticles/PeerLearningDays.md') as f:
      reference = f.read()
  
  print(question_answer('When are PLDs?', reference))
  $ ./0-main.py
  on - site days from 9 : 00 am to 3 : 00 pm
  $
  
```






**Repo:**


* GitHub repository: `holbertonschool-machine_learning`
* Directory: `supervised_learning/qa_bot`
* File: `0-qa.py`










 Help
 




×
#### Students who are done with "0\. Question Answering"


















**0/8** 
pts









### 1\. Create the loop




 mandatory
 






Create a script that takes in input from the user with the prompt `Q:` and prints `A:` as a response. If the user inputs `exit`, `quit`, `goodbye`, or `bye`, case insensitive, print `A: Goodbye` and exit.



```
$ ./1-loop.py
  Q: Hello
  A:
  Q: How are you?
  A:
  Q: BYE
  A: Goodbye
  $
  
```






**Repo:**


* GitHub repository: `holbertonschool-machine_learning`
* Directory: `supervised_learning/qa_bot`
* File: `1-loop.py`










 Help
 




×
#### Students who are done with "1\. Create the loop"


















**0/6** 
pts









### 2\. Answer Questions




 mandatory
 






Based on the previous tasks, write a function `def answer_loop(reference):` that answers questions from a reference text:


* `reference` is the reference text
* If the answer cannot be found in the reference text, respond with `Sorry, I do not understand your question.`



```
$ cat 2-main.py
  #!/usr/bin/env python3
  
  answer_loop = __import__('2-qa').answer_loop
  
  with open('ZendeskArticles/PeerLearningDays.md') as f:
      reference = f.read()
  
  answer_loop(reference)
  $ ./2-main.py
  Q: When are PLDs?
  A: from 9 : 00 am to 3 : 00 pm
  Q: What are Mock Interviews?
  A: Sorry, I do not understand your question.
  Q: What does PLD stand for?
  A: peer learning days
  Q: EXIT
  A: Goodbye
  $
  
```






**Repo:**


* GitHub repository: `holbertonschool-machine_learning`
* Directory: `supervised_learning/qa_bot`
* File: `2-qa.py`










 Help
 




×
#### Students who are done with "2\. Answer Questions"


















**0/7** 
pts









### 3\. Semantic Search




 mandatory
 






Write a function `def semantic_search(corpus_path, sentence):` that performs semantic search on a corpus of documents:


* `corpus_path` is the path to the corpus of reference documents on which to perform semantic search
* `sentence` is the sentence from which to perform semantic search
* Returns: the reference text of the document most similar to `sentence`



```
$ cat 3-main.py
  #!/usr/bin/env python3
  
  semantic_search = __import__('3-semantic_search').semantic_search
  
  print(semantic_search('ZendeskArticles', 'When are PLDs?'))
  $ ./ 3-main.py
  PLD Overview
  Peer Learning Days (PLDs) are a time for you and your peers to ensure that each of you understands the concepts you've encountered in your projects, as well as a time for everyone to collectively grow in technical, professional, and soft skills. During PLD, you will collaboratively review prior projects with a group of cohort peers.
  PLD Basics
  PLDs are mandatory on-site days from 9:00 AM to 3:00 PM. If you cannot be present or on time, you must use a PTO. 
  No laptops, tablets, or screens are allowed until all tasks have been whiteboarded and understood by the entirety of your group. This time is for whiteboarding, dialogue, and active peer collaboration. After this, you may return to computers with each other to pair or group program. 
  Peer Learning Days are not about sharing solutions. This doesn't empower peers with the ability to solve problems themselves! Peer learning is when you share your thought process, whether through conversation, whiteboarding, debugging, or live coding. 
  When a peer has a question, rather than offering the solution, ask the following:
  "How did you come to that conclusion?"
  "What have you tried?"
  "Did the man page give you a lead?"
  "Did you think about this concept?"
  Modeling this form of thinking for one another is invaluable and will strengthen your entire cohort.
  Your ability to articulate your knowledge is a crucial skill and will be required to succeed during technical interviews and through your career. 
  $
  
```






**Repo:**


* GitHub repository: `holbertonschool-machine_learning`
* Directory: `supervised_learning/qa_bot`
* File: `3-semantic_search.py`










 Help
 




×
#### Students who are done with "3\. Semantic Search"


















**0/7** 
pts









### 4\. Multi\-reference Question Answering




 mandatory
 






Based on the previous tasks, write a function `def question_answer(coprus_path):` that answers questions from multiple reference texts:


* `corpus_path` is the path to the corpus of reference documents



```
$ cat 4-main.py
  #!/usr/bin/env python3
  
  question_answer = __import__('4-qa').question_answer
  
  question_answer('ZendeskArticles')
  $ ./4-main.py
  Q: When are PLDs?
  A: on - site days from 9 : 00 am to 3 : 00 pm
  Q: What are Mock Interviews?
  A: help you train for technical interviews
  Q: What does PLD stand for?
  A: peer learning days
  Q: goodbye
  A: Goodbye
  $
  
```






**Repo:**


* GitHub repository: `holbertonschool-machine_learning`
* Directory: `supervised_learning/qa_bot`
* File: `4-qa.py`










 Help
 




×
#### Students who are done with "4\. Multi\-reference Question Answering"


















**0/10** 
pts









[Previous project](/projects/2422)  



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





0\. `Question Answering`



1\. `Create the loop`



2\. `Answer Questions`



3\. `Semantic Search`



4\. `Multi-reference Question Answering`










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





