Extraction of causal relations from text


First run preprocess.py

Then run train.py

Finally run test.py to test sentences written in my_ctext2.txt

pip install pydot

pip install pydotplus

sudo apt-get install graphviz
 
# Modeling

We want to model the relations between the macro-economic factors and explore these relations before and after the COVID outbreak
with an implementation of a Bayesian Network based on a causality detection modeling that will be
presented in details with its different steps in this part.

## Causality detection

Causality detection is a nascent field in Natural Language Processing domain. It can have many
forms ranging from using explicit pattern-matching rules to more Machine Learning oriented results.
The entities involved in the detection of causality can vary from one use case to another. We can
find causal relations between sentences, events, or entities in a sentence.
In this work, we will consider the previous data to extract causal relationships between macroeco-
nomic entities from millions of financial news articles. These relations would serve as the main brick
to the construction of the Bayesian Network.

## CNN for causality detection

Convolutional Neural Networks (CNN) were initially developed in the field of neural network image
processing, where they achieved break-through results in classifying objects from predefined cate-
gories.
In recent years, CNN have shown ground-breaking performance in various NLP tasks. One particular
task is sentence classification, i.e., classifying short phrases (i.e., roughly 20–50 tokens), from a set of
pre-defined categories. In the next paragraphs we will explain how CNN can be applied to classify
causal sentences.

## Data

On this step we will use SemEval-2 Task8 by [Hendrickx and Kim, 2010] which focuses on multiple-
way classification of the semantic relationships between nominal pairs. It is a testbed for automatic
classification of semantic relations.
23This dataset consists of 8000 sentences that have been annotated in accordance with SemEval-2 Task
8’s guidelines. Some sentences have been reused from SemEval-1 Task4 (Classification of Semantic
Relations between Nominals), For this work, the rest have been gathered from the Internet. All have
been annotated in accordance with the new relation definitions included in this data release.

## Data format

The format of the data is a Cause-Effect (CE) an event or object that leads to an effect as it is
illustrated by the following examples:

Example:

1541 "The risks to housing and general corporate <e1>profits</e1> from <e2>inflation</e2>
were quite clear and the defensive strategy over the long term has paid dividends."
Cause-Effect(e2,e1)
Comment:

The first line contains the sentence itself inside quotation marks, preceded by a numerical identifier.
Each sentence is annotated with three pieces of information:

<ul>
<li> The sentence’s two entity mentions are designated as e1 and e2; the numbering simply reflects
the order of the mentions. The "base NP," which may be less than the entire NP designating
the entity, corresponds to the tag’s span. </li> 
<li>  If one of the semantic relations 1–9 exists between e1 and e2, the relation’s name and the order
in which e1 and e2 fill out the relation arguments are given as labels for the sentence.
For example, Cause-Effect(e1,e2), for instance, denotes that e1 is the Cause and e2 is the
Effect, but Cause-Effect(e2,e1) denotes that e2 is the Cause and e1 is the Effect. if no family
members. If none of the relations 1-9 holds, the sentence is labelled "Other". Therefore, a total
of 19 labels are possible. </li>
<li>  The test data’s comments won’t be disclosed. It should be noted that the test release will be
styled similarly but will not include lines for the related label or the comment. </li>
</ul>

Note that the test release will have a similar layout but will not have lines for the related label or
the comment.

## Methodology

A general workflow for model training and evaluation is shown below.

Causality relation classification: Training and Evaluation pipeline:

![image](https://user-images.githubusercontent.com/47029962/210543062-b6a00525-a670-4abf-8fd2-70913d7e6311.png)

## Preprocessing

Data preprocessing is done to transform raw data into a format that comply with our requirements
to ensure and enhance performance. The steps are as follows:

<ul>
<li> Keeping only the cause and effect relation classes and labeling the rest of the 8 relations classes
(Entity-Origin, Entity-Destination, Message-Topic,...) as Other. </li>
<li> Mapping of the labels to integers:
’Cause-Effect(e1,e2)’:1
’Cause-Effect(e2,e1)’:2
’Other’:0 </li>
<li> Applying word embeddings:

To capture the semantic and syntactic meaning of a word, word embedding is done using the
pretrained Extended Dependency Based Skip-gram which is an unsupervised learning of
word representations method for training word embeddings using structural information from
dependency graphs as described in [Komninos and Manandhar, 2016]. In addition to standard
word embeddings, it produces embeddings of dependency context features which were found
to be useful features in several sentence classification as largely improved the performance of
the classifiers for semantic relation identification. The idea behind the pretrained Extended
Dependency Based Skip-gram is to predict as output the context neighbors words in a sentence
from the target word given as input based on using other contextual features, such as contexts
from dependency graphs of sentences. and finally each word is fed into a model as a one hot
encoding vector. It is slower than the algorithms that predict the target word from the con-
text words but skip-gram does a better job predicting infrequent words and is more suitable
for large corpus with higher dimensions. Furthermore, the dependency context embeddings
improve performance with all tested classifiers. </li>
</ul>

Architecture:

The words embeddings Skip-gram model architecture:

![image](https://user-images.githubusercontent.com/47029962/210543594-c351e300-4f39-4ee7-8afd-0dc42a669ddb.png)

Example:

Consider the following example case:

"NLP process human language as text or speech to make computers similar to humans."
The word "process" will be given and we’ll try to predict the rest of context words "NLP", "human
language", "speech", ... given "process" is at position 0 . We do not predict common or stop
words such as "the" .

The words embeddings Skip-gram model sentence sample:

![image](https://user-images.githubusercontent.com/47029962/210543874-962866de-8c41-4959-b2ab-3aa9bbf2ee6f.png)

This pretrained wordembeddings includes:
<ul>
<li> Word embeddings were trained on Wikipedia August 2015. </li>
<li> Includes embeddings of words and dependency contexts appearing more than 100 times in the
corpus. </li>
<li> The dependency types used are [Universal_Dependencies, 2015] which is a framework for con-
sistent annotation of grammar (parts of speech, morphological features, and syntactic depen-
dencies) across different human languages. </li>
<li> Inverse relations are encoded with the string "_inv_" between the dependency type and the
word. </li>
<li> Training corpus size: 2 billion words </li>
<li> Word vocabulary size: 222,496 </li>
<li> Dependency context vocabulary size: 1,253,524 </li>
<li> Embedding dimensionality: 300 </li>
<li> Training was done by applying negative sampling with 15 negative samples per target-context
pair for 10 iterations over the entire corpus using stochastic gradient descent. </li>
<li> The advantage from using pretrained word embeddings is to increase performance compared
to learning embeddings from scratch by:
<ol>
<li>  Reusing the best trained hyper parameters. </li>
<li>  Save resources since Softmax function is computationally expensive. </li>
<li>  Save time since the time required for training this algorithm is high. </li>
</ol>
</li>
</ul>




