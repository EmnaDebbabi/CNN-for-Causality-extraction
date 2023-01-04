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
<li> Padding the sequences:
 
The lengths of the sequences vary. A machine learning model is usually fed a series of the
same length. As a result, you must ensure that all sequences are the same length. This is
accomplished via padding sequences. Shorter sequences will be padded with zeros, while longer
ones will be truncated. As a result, you’ll need to declare the truncation and padding types. </li>
 <li> Split data into train, test and validation batches. </li>
 <li> Set class weights using Sklearn class weight since the SemEval2010 task8 for causality classifi-
cation has imbalanced classes: (6996 "Other" relation, 659 "Cause-Effect(e2,e1)" effect relation,
344 "Cause-Effect(e1,e2)" causality relation ) in order to force the algorithm to treat every
instance of class 0, 1 and 2 as its adjusted class weight indicates:
  class_weight = 0: 0.38111572, 1: 7.75193798, 2: 4.04653515 </li>
</ul>

## Model training

The next step is to use the words embeddings layer in a Keras model. We use the simple Convo-
lutional Neural Network of [Daojian Zeng and Zhao, 2014] that has been shown to perform well in
multiple sentence classification tasks. Let’s define the model as follows:


<ul>
  
 <li> As the top layer, the embedding layer. </li>
 <li>  Then the network’s input is a sentence matrix X formed by concatenating k-dimensional word
embeddings. </li>
 <li>  Then a convolutional filter W ∈ R h∗k is applied to every possible sequence of length h to get a
feature map: 
  
  ![image](https://user-images.githubusercontent.com/47029962/210577397-78ef8685-c80a-42ec-9f43-4acf97397156.png)

  </li>
 <li> Followed by a max-over-time pooling operation to get the feature with the highest value:
 
  ![image](https://user-images.githubusercontent.com/47029962/210577569-215eb614-86fd-4397-9a7e-be119596549a.png)

 </li>
 <li> The last layer is in charge of the output; it performs classification by concatenating the pooled
features from all previous layers and passing them to a fully connected softmax layer. Multiple
filters are used by the network, each covering a different size window in the text and using a
different sequence length. Performance with hyperparameter optimization: Stochastic dropout
with p = 0.25 on the penultimate layer, 100 filters for each filter region with filter regions of
width 3. Optimization is performed with Adam algorithm on minibatches of size 64 and a
sparse categorical crossentropy loss function. </li>
 <li> Architecture: 
  
  CNN model for text classification architecture:
 
  ![image](https://user-images.githubusercontent.com/47029962/210581669-9d42a88a-c759-4443-96f2-d7c35800ef0a.png)

 </li>
 </ul>
 
## Prediction

Now the CNN model is created and fitted with trained data, and can be used to make a prediction.
A final model can be saved, and then loaded again and reconstructed. The reconstructed model has
already been compiled and has retained the optimizer state, so that training can resume with either
historical or new data. We will apply the same Skip-gram words embeddings used before on the
test data then we will predict the test data using model predict of Keras. Trained classifier makes
predictions on the 2717 samples from the SemEval-2 Task8 test set.

## Conclusion

In this part, we presented the convolutional deep neural network (CNN) for relation classification
and its different steps to illustrate how Bayesian Network modeling can be used. In the next part
we will evaluate our model by testing it on a real data and also by explaining how we are going to
implement it.

## Evaluation

In this part, we outline the methods for assessing how effectively our CNN causality detection
model generalizes to new, unexplored data. and how it is capable to handle the problem of causality
classification.

To evaluate our CNN causality detection model, confusion matrix of true labels vs predicted labels,
accuracy and loss graph are plotted.
We were able to achieve an accuracy, which presents the number of correct predictions made as a
ratio of all predictions made, of 87.9% over SemEval-2 Task8 test data that measures the effectiveness
of CNN model in correctly predicting both Cause, Effect and Other relations between two entities
in sentences.

F1 score macro and micro for each epoch in keras are computed:

For any given metric, micro- and macro-averages will compute somewhat differently, and as a result,
their interpretation varies. For each class, a macro-average will independently calculate the metric
and then take the average. In contrast, a micro-average will add up the contributions from each class
to determine the average measure. If there may be a class imbalance in a multi-class classification
system, macro-average is preferred, which is our case we have many more examples of the class
"Other" relation than of other classes "Cause" and "Effect" relation.
Also to compare our results with those obtained in previous studies, we adopt the macroaveraged
F1-score and also account for directionality into account in our following experiments; we obtained
val_f1: 0.894.

## Confusion matrix

Based on the 3x3 confusion matrix in our case and as shown below; the columns are the predictions
and the rows are the actual values. The main diagonal (2364, 121, 177) gives the correct predictions.
That is, the cases where the actual values and the model predictions are the same.

The first row are the "Other" relation. The model predicted 2364 of these correctly and incorrectly predicted 5 of the "Other" relation to be "Cause" relation and 20 of the "Other" relation to be "Effect" relation.

Looking at the "Other" relation column, of the 2391 "Other" relation predicted by the model (sum of
column "Other"), 2364 were actually "Other" relation, while 10 were "Cause" relation incorrectly predicted to be "Other" relation and 17 were "Effect" relation incorrectly predicted to be "Other" relation.

Analogous interpretations apply to the other columns and rows.

![image](https://user-images.githubusercontent.com/47029962/210589650-083d6d68-beeb-48aa-8919-82a6af70dbe0.png)

Confusion Matrix:

![image](https://user-images.githubusercontent.com/47029962/210589803-65b13b92-6de6-4623-a60c-2d126e95a6dd.png)


## Comparaison CNN with RNN and HAN

To evaluate more effectively the CNN model we compared it with the Recurrent Neural Network
(RNN) and Hierarchical Attention Network (HAN) based on three different dataset:


Different used datasets details:

![image](https://user-images.githubusercontent.com/47029962/210590123-b5c564ef-de5f-4301-9c78-784250a7d910.png)

Results:

The results of CNN, RNN and HAN models:

![image](https://user-images.githubusercontent.com/47029962/210590376-799ccc3d-ecb2-4135-bf3c-bc92db818b5b.png)

Accuracy graph of CNN, RNN and HAN models:

![image](https://user-images.githubusercontent.com/47029962/210604338-a4e6ff0d-cab7-4101-aee1-fa1b70efe402.png)

<ul>
<li> Based on the above plots, CNN has achieved good validation accuracy with high consistency,
also RNN and HAN have achieved high accuracy, but they are not that consistent throughout
all the datasets. </li>
<li> RNN was found to be the worst architecture to implement for production ready scenarios.</li>
<li> CNN model has outperformed the other two models (RNN and HAN) in terms of training time.
However, HAN can perform better than CNN and RNN if we have a huge dataset. </li>
<li> For dataset1 and dataset2 where the training samples are big, HAN has achieved the best
validation accuracy while when the training samples are very low, then HAN han not performed
that good (dataset3). </li>
<li> When training samples are low (dataset3) CNN has achieved he best validation accuracy. </li>
 </ul>

## Concluding remarks and future scope

In this part of project, we presented CNN for relation classification to illustrate how Bayesian Network
modeling can be used, we utilize the algorithm to generate and fit a graph using TextReveal Data
based on the following entities: "dollar", "inflation", "oil", "gold", "consumption", "import", "export",
"retail" and "aviation". For these entities, we recovered around 5.5 million articles ranging from
19-07-2016 to 26-07-2021. From these news articles, we were able to extract around 38000 causal
relationships. The goal is to analyze the relationships between these variables both before and
after COVID based on their causality classification and direction detection. The Bayesian Network
obtained before and after "01-01-2020" is presented in the following graph.

Bayesian Network before and after COVID example of final results:

![image](https://user-images.githubusercontent.com/47029962/210607050-e47309ca-34e9-489e-9242-9eb7087d6803.png)

An association between two events—a cause and an effect—is known as a causal relationship. Effect
is what the cause produces, and cause is what the effect is.

Ex."Inflation might result from a spike in demand for goods and services as consumers are willing to
pay more for the product." Here cause is "A surge in demand for products and services" and effect
is "inflation". In the present work we focused on the detection and extraction of Causal Relations
from news articles texts. From the point of view of detecting Causal Relations, we can follow this
distinctions:

<ul>
<li> A causality is either marked or unmarked depending on whether it is indicated by a certain
language unit. "Inflation can be a concern because it reduces the value of money saved today."
is marked; "Be careful. It’s unstable" isn’t. </li>
<li> Ambiguity: If the mark always denotes a causal relationship, it is clear (e.g. "because"). If it
signals sometimes a causation, it is ambiguous (e.g. "since" ). </li>
<li> Explicit or implicit: Causation can be either explicit or implicit; if one or both reasons are
absent, it is implicit. "Inflation erodes a consumer’s purchasing power after interfering with his
ability to retire." is explicit; "Inflation killing small business." is implicit, since the effect, small
business death, is not explicitly stated. </li>
<ul>
 
 CNN model has been selected for classification because of its advantages:
 
 <ul>
  <li> Suitable for large and noisy data for example for 1M doc from news, blogs and tweets,...</li>
<li> No text preprocessing </li>
<li> Good handling of misspelled or out-of-vocabulary words </li>
  <li>Fast training </li>
  <li>Small model size and low memory usage </li>
 </ul>
 
 CNN has shown good results for the marked and explicit causation and has shown limits with
the classification of the unmarked and implicit causation. As a perspective to our work, causality
extraction can be improved
 
 ## Conclusion
 
In this part we evaluated our CNN in terms of speed ,accuracy and efficiency by proving that the
CNN model is capable to handle the problem of causality classification on textual data to model relationships be-
tween macro-economic factors with Bayesian Network and based on large financial news textual data
extracted. We used Convolutional Neural Networks (CNN) for Sentence Classification, to extract causal relations from financial news and to build a database of causalities. Next
the project main goal is to use these causalities to feed an optimization algorithm to build and fit a
Bayesian Network to compare macro-economic before and after the COVID outbreak to understand
what factors were impacted.
 
As an extension of our work, we intend to improve causality extraction, direction detection us-
ing rule-based methods to ensure better precision in producing Bayesian network-based economic
and financial insights.
