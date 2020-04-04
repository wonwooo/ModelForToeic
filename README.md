
### Corret Answer Rate with Only Pre-trained BERTForMaskedLM : 83.8% 
### After finetuning BERT through the proposed method : 87.8%

This project started by referring to the project of [graykode](<https://github.com/graykode/toeicbert>) who solved TOEIC Part 5(Sentence with blank problem) with pytorch-pretrained-BERT model(Not finetuned).
This project was done to increase the correct answer rate for the TOEIC Part5 problems by finetuning pretrained-BERT.

We collected a total of 6100 Part5 problems, and used 85% (5185) and 15% (915) questions for training and testing.
You can see whole process in this project [here](<https://github.com/wonwooo/ModelForToeic/blob/master/Toeic_Bert.ipynb>).


## TOEIC Part 5 : Blank sentence problems

There are two types in  TOEIC Part 5 as below:
#### Type 1 : Grammer
```
Question : The marketing seminar is being [ ? ] from August 8th through the 11th at Rupp Convention Center.
    a) held
    b) holds
    c) holding
    d) hold
```

#### Type 2 : Vocabulary
```
Question : THe appointment will bring a great deal of [ ? ].
    a) prestige
    b) testimony
    c) willpower
    d) virtuosity    
```
We have to choose best one of the four candidates given in the problem. 

## 1. Prerained BERT For Masked Language Model

We first measured the performance of the pretrained BERT using the transformer package provided by Huggingface. Here is an example of the problem we used for Test.

```
{
    '1' :{'question': 'His allergy symptoms _ with the arrival of summer.',
  'answer': 'worsen',
  '1': 'bad',
  '2': 'worse',
  '3': 'worst',
  '4': 'worsen'},

 '2' : {'question': 'He told us that some fans lined up outside of the box office to _ a ticket for the concert.',
  'answer': 'purchase',
  '1': 'achieve',
  '2': 'purchase',
  '3': 'replace',
  '4': 'support'}
}
```
To solve this blank problems with huggingface's [pytorch-pretrained-BERT model](<https://github.com/huggingface/pytorch-pretrained-BERT>), the method suggested by  [graykode](<https://github.com/graykode/toeicbert>) was borrowed.

As a result of the test, Pretrined BertForMaskedLM already showed a correct answer rate of 83.8%.
We tried to finetune Pretrined BertForMaskedLM with 5185 training sets. But it was no different from Bert's original pretraining task, so there was no improvement in the correct answer rate for the test problem. The dataset we created to Finetune the MaskedLM model is as follows.

| Sentence(X)     | Output(Y) |
| :-------------: |  :--------------: |
| The marketing seminar is being [ ? ] from August 8th through the 11th at Rupp Convention Center. |    held (Correct answer)   |
| The appointment will bring a great deal of [ ? ] |    prestige (Correct answer)   |


## 2. Grammar learning specialized for TOEIC Part5.

Therefore, we devised a method to increase the correct answer rate by learning the features of the grammatical part by slightly changing the task.
First, a model with a linear layer for binary classification was used on the pre-trained Bert.(huggingface's [BertForSequenceClassification](<https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification>))

To finetune the linear classification model, we created four trainig data in one of the following problems.

```
Question : The marketing seminar is being [ ? ] from August 8th through the 11th at Rupp Convention Center.
    a) held
    b) holds
    c) holding
    d) hold
``` 

| Sentence(X)     | Output(Y) |
| :-------------: |  :--------------: |
| The marketing seminar is being `held` from August 8th through the 11th at Rupp Convention Center. |  True(1)  |
| The marketing seminar is being `holds` from August 8th through the 11th at Rupp Convention Center. |    False(0)  |
| The marketing seminar is being `holding` from August 8th through the 11th at Rupp Convention Center. |    False(0)  |
| The marketing seminar is being `hold` from August 8th through the 11th at Rupp Convention Center. |    False(0)  |

We got 20744 training samples from 5186 questions and tried to train(finetune) BertForSequenceClassification model.


## 3. Solving Part5 problems with finedtuned BertForSequenceClassification model

To solve Part5 questions with the finetuned Classifier model, we should input each full sentence that blank is filled blank with 4 candidates in question. Our input sentence X and model's output is shown below.

```
Question : The marketing seminar is being [ ? ] from August 8th through the 11th at Rupp Convention Center.
    a) held
    b) holds
    c) holding
    d) hold
``` 


|  | Input(X)     | Output(Y) |
| :-------------: | :-------------: |  :--------------: |
|X<sub>1</sub>| The marketing seminar is being `held` from August 8th through the 11th at Rupp Convention Center. | BertForSeqClassification(X<sub>1</sub>) = [logit<sub>True</sub>(X<sub>1</sub>) , logit<sub>False</sub>(X<sub>1</sub>)]|
|X<sub>2</sub>| The marketing seminar is being `holds` from August 8th through the 11th at Rupp Convention Center. |BertForSeqClassification(X<sub>2</sub>) = logit<sub>True</sub>(X<sub>2</sub>) , logit<sub>False</sub>(X<sub>2</sub>)]|
|X<sub>3</sub>| The marketing seminar is being `holding` from August 8th through the 11th at Rupp Convention Center. |    BertForSeqClassification(X<sub>3</sub>) = logit<sub>True</sub>(X<sub>3</sub>) , logit<sub>False</sub>(X<sub>3</sub>)] |
|X<sub>4</sub>| The marketing seminar is being `hold` from August 8th through the 11th at Rupp Convention Center. |    BertForSeqClassification(X<sub>4</sub>) = logit<sub>True</sub>(X<sub>4</sub>) , logit<sub>False</sub>(X<sub>4</sub>)] |

If the above X is tokenized and input to the Finetuned BERTForSequenceClassification model, the model outputs the logit about the true and false of sentence X respectively. Finally, we predict candidates with the highest logit<sub>True</sub> as correct answer .

<p align="center"><img width="500" src="https://github.com/woopal/ModelForToeic/blob/master/eq1.PNG"/></p>
