Training extended dependency skip-gram embeddings as described in [1].


Given a dependency parsed corpus create a training file with the following format.
For every node of the dependency graph create two lines in the training file:
	word dependency_context1 dependency_context2 ...
	word word_context1 word_context2 ...
word: target word in the dependency graph.
dependency_context: words with a dependency relation to the target node (e.g. compound_programming, compound_inv_language)
word_context: words within distance 1 in the graph.


word2vec_ext.c:
Modification of the original word2vec code (https://word2vec.googlecode.com/svn/trunk/)
for training skip-gram embeddings with negative sampling.
Creates target-context pairs for all tokens appearing in the same line of the input file.
The extended dependency skip-gram model can be trained by providing a training file
constructed in the way described above.
It is recommended to provide a separate vocabulary file with the actual token counts from
the corpus, since constructing the vocabulary from the training file will count duplicates.

corenlp2trainf.py:
Script for constructing an extended dependency skip-gram training file and the corresponding
vocabulary file from dependency parsed sentences in Stanford's CoreNLP text output format.
(http://stanfordnlp.github.io/CoreNLP/).

wiki_sample_parsed: A small sample of sentences from the Wikipedia August 2015 dump
(https://dumps.wikimedia.org/enwiki/20150805/) parsed with Stanford's CoreNLP.

Example usage:

python corenlp2trainf.py wiki_sample_parsed wiki_sample_contexts wiki_sample_voc 0

./word2vec_ext -train wiki_sample_contexts -read-vocab wiki_sample_voc -output wiki_sample_vecs -output-ctx wiki_sample_cvecs -min-count 0 -iter 1 -threads 2 -negative 15 -sample 0 -size 200  

 
[1] Alexandros Komninos and Suresh Manandhar. 2016. Dependency Based Embeddings for Sentence Classification Tasks.
In Proceedings of NAACL.
