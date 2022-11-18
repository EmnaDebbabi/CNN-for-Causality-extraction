from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import gzip
import os
import sys
import pickle as pkl
from utils import *
import keras
from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, concatenate
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D
from keras.regularizers import Regularizer
from keras.preprocessing import sequence
import tensorflow as tf



class CausalityCNN():
    
    def __init__(self,folder=None,files=None,embbed_extvec=True,classesLabelling=False):
        """
        Preprocesses the files/train.txt and files/test.txt files.
        Parameters
        ----------
        files :  
            list contains train and test data path

        Returns
        -------
        str
            new_file: train data
        The dependency based embeddings by Levy et al.. Download from the website and change 
        the embeddingsPath variable in the script to point to the unzipped deps.words file.
        """             

        def ClassesLabelling (old_file=None,new_file=None):
            """
            Keep relation type 'cause_effect' and 'Other' 

            Parameters
            ----------
            old_file : str 
                train data path

            Returns
            -------
            str
                new_file: train data 
            """
            if (old_file==None):  
                old_file = open('data/train.txt', 'r')
                
            if (new_file==None):
                new_file = open('data/ctrain.txt', 'w+')

            labelsMapping = {'Other':0, 
                            'Message-Topic(e1,e2)':0, 'Message-Topic(e2,e1)':0, 
                            'Product-Producer(e1,e2)':0, 'Product-Producer(e2,e1)':0, 
                            'Instrument-Agency(e1,e2)':0, 'Instrument-Agency(e2,e1)':0, 
                            'Entity-Destination(e1,e2)':0, 'Entity-Destination(e2,e1)':0,
                            'Cause-Effect(e1,e2)':0, 'Cause-Effect(e2,e1)':0,
                            'Component-Whole(e1,e2)':0, 'Component-Whole(e2,e1)':0,  
                            'Entity-Origin(e1,e2)':0, 'Entity-Origin(e2,e1)':0,
                            'Member-Collection(e1,e2)':0, 'Member-Collection(e2,e1)':0,
                            'Content-Container(e1,e2)':0, 'Content-Container(e2,e1)':0}

            for l in old_file:
                arr = l.split('\t')
                relation_type = arr[0]
                if (relation_type.startswith('Cause-Effect')):
                    new_file.write(l)
                else:
                    new_file.write('Other')
                    new_file.write('\t')
                    new_file.write(arr[1])
                    new_file.write('\t')
                    new_file.write(arr[2])
                    new_file.write('\t')
                    new_file.write(arr[3])
            print ('classes labelling done')
        if (classesLabelling==True):
            ClassesLabelling ()
        else:
            pass
        if (folder==None) and (folder==None):            
            folder = 'data/'
            files = [folder+'ctrain.txt', folder+'ctest.txt']
        #Mapping of the labels to integers
        labelsMapping = {'Other':0, 'Cause-Effect(e1,e2)':1, 'Cause-Effect(e2,e1)':2}
        words = {}
        maxSentenceLen = [0,0]
        minDistance = -30
        maxDistance = 30
        folder='data/'
        outputFilePath = folder + 'causal-relations.pkl.gz'

        for fileIdx in range(len(files)):
            file = files[fileIdx]
            for line in open(file):
                
                splits = line.strip().split('\t')
                label = splits[0]
                sentence = splits[3]        
                tokens = sentence.split(" ")
                maxSentenceLen[fileIdx] = max(maxSentenceLen[fileIdx], len(tokens))
                for token in tokens:
                    words[token.lower()] = True   
        print("Max Sentence Lengths: ", maxSentenceLen)


        def embeddings_extvec(folder,files):   
            outputFilePath = folder + 'causal-relations.pkl.gz'
            #Download English word embeddings from here https://www.cs.york.ac.uk/nlp/extvec/
            embeddingsPath = folder + 'wiki_extvec.gz'
            # :: Read in word embeddings ::
            word2Idx = {}
            wordEmbeddings = []

            # :: Downloads the embeddings from the York webserver ::
            if not os.path.isfile(embeddingsPath):
                basename = os.path.basename(embeddingsPath)
                if basename == 'wiki_extvec.gz':
                       print("Start downloading word embeddings for English using wget ...")
                       #os.system("wget https://www.cs.york.ac.uk/nlp/extvec/"+basename+" -P embeddings/")
                       os.system("wget https://public.ukp.informatik.tu-darmstadt.de/reimers/2017_english_embeddings/"+basename+" -P embeddings/")
                else:
                    print(embeddingsPath, "does not exist. Please provide pre-trained embeddings")
                    exit()

            # :: Load the pre-trained embeddings file ::
            fEmbeddings = gzip.open(embeddingsPath, "r") if embeddingsPath.endswith('.gz') else open(embeddingsPath, encoding="utf8")    
            print("Load pre-trained embeddings file")
            for line in fEmbeddings:
                split = line.decode('utf-8').strip().split(" ")
                word = split[0]

                if len(word2Idx) == 0: #Add padding+unknown
                    word2Idx["PADDING_TOKEN"] = len(word2Idx)
                    vector = np.zeros(len(split)-1) #Zero vector vor 'PADDING' word
                    wordEmbeddings.append(vector)

                    word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
                    vector = np.random.uniform(-0.25, 0.25, len(split)-1)
                    wordEmbeddings.append(vector)

                if word.lower() in words:
                    vector = np.array([float(num) for num in split[1:]])
                    wordEmbeddings.append(vector)
                    word2Idx[word] = len(word2Idx)


            wordEmbeddings = np.array(wordEmbeddings)
            print("Embeddings shape: ", wordEmbeddings.shape)
            print("Len words: ", len(words))

            return wordEmbeddings,word2Idx

        if embbed_extvec:
            wordEmbeddings,word2Idx=embeddings_extvec(folder,files)
            # :: Create token matrix ::
            vectorizer = Vectorizer(word2Idx, labelsMapping, minDistance, maxDistance, max(maxSentenceLen))
            train_set = vectorizer.vectorizeInput(files[0])
            test_set = vectorizer.vectorizeInput(files[1])

            data = {'wordEmbeddings': wordEmbeddings, 'word2Idx': word2Idx, 
                    'train_set': train_set, 'test_set': test_set, 'labels_mapping': labelsMapping, 'max_sentence_length': max(maxSentenceLen), 'min_distance': minDistance, 'max_distance': maxDistance}

            f = gzip.open(outputFilePath, 'wb')
            pkl.dump(data, f)
            f.close()

            print("Data stored in pkl folder") 
        else:
            pass 

    def trainModel(trainmodel=False):
        """
        This is a CNN for relation classification within a sentence. The architecture is based on:

        Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou and Jun Zhao, 2014, Relation Classification via Convolutional Deep Neural Network

        Performance (without hyperparameter optimization):
        Accuracy: 0.7943
        Macro-Averaged F1 (without Other relation):  0.7612

        Performance Zeng et al.
        Macro-Averaged F1 (without Other relation): 0.789


        Code was tested with:
        - Python 3.6
        - Theano 1.0.5 & TensorFlow 2.5.0
        - Keras 2.5.0
        """
        if trainmodel==False:
            model = load_model('model/causal_rel_model.h5')
            print('model loaded')
            return model
        else:
            batch_size = 64
            nb_filter = 100
            filter_length = 3
            hidden_dims = 100
            nb_epoch = 30
            position_dims = 50
            print("Load dataset")
            f = gzip.open('data/causal-relations.pkl.gz', 'rb')
            data = pkl.load(f)
            f.close()
            embeddings = data['wordEmbeddings']
            yTrain, sentenceTrain, positionTrain1, positionTrain2 = data['train_set']
            yTest, sentenceTest, positionTest1, positionTest2  = data['test_set']

            max_position = max(np.max(positionTrain1), np.max(positionTrain2))+1

            n_out = max(yTrain)+1
            #train_y_cat = np_utils.to_categorical(yTrain, n_out)
            max_sentence_len = sentenceTrain.shape[1]
            print("sentenceTrain: ", sentenceTrain.shape)
            print("positionTrain1: ", positionTrain1.shape)
            print("yTrain: ", yTrain.shape)
            print("sentenceTest: ", sentenceTest.shape)
            print("positionTest1: ", positionTest1.shape)
            print("yTest: ", yTest.shape)
            print("Embeddings: ",embeddings.shape)
            #This is a CNN for relation classification within a sentence.
            words_input = Input(shape=(max_sentence_len,), dtype='int32', name='words_input')
            words = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], trainable=False)(words_input)

            distance1_input = Input(shape=(max_sentence_len,), dtype='int32', name='distance1_input')
            distance1 = Embedding(max_position, position_dims)(distance1_input)

            distance2_input = Input(shape=(max_sentence_len,), dtype='int32', name='distance2_input')
            distance2 = Embedding(max_position, position_dims)(distance2_input)

            output = concatenate([words, distance1, distance2])

            output = Convolution1D(filters=nb_filter,
                                    kernel_size=filter_length,
                                    padding='same',
                                    activation='tanh',
                                    strides=1)(output)

            # standard max over time pooling
            output = GlobalMaxPooling1D()(output)

            output = Dropout(0.25)(output)
            output = Dense(n_out, activation='softmax')(output)
            model = Model(inputs=[words_input, distance1_input, distance2_input], outputs=[output])
            model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam', metrics=['accuracy'])
            model.summary()

            print("Start training")
       

        max_prec, max_rec, max_acc, max_f1 = 0,0,0,0

        def getPrecision(pred_test, yTest, targetLabel):
            #Precision for non-vague
            targetLabelCount = 0
            correctTargetLabelCount = 0

            for idx in range(len(pred_test)): 
                if pred_test[idx] == targetLabel:
                    targetLabelCount += 1

                    if pred_test[idx] == yTest[idx]:
                        correctTargetLabelCount += 1

            if correctTargetLabelCount == 0:
                return 0

            return float(correctTargetLabelCount) / targetLabelCount

        def predict_classes(prediction):
            return prediction.argmax(axis=-1)

        for epoch in range(nb_epoch):       
            model.fit([sentenceTrain, positionTrain1, positionTrain2], yTrain, batch_size=batch_size, verbose=True,epochs=1)   
            pred_test_ini = model.predict([sentenceTest, positionTest1, positionTest2], verbose=False)
            pred_test = predict_classes(pred_test_ini)

            dctLabels = np.sum(pred_test)
            totalDCTLabels = np.sum(yTest)

            acc =  np.sum(pred_test == yTest) / float(len(yTest))
            max_acc = max(max_acc, acc)
            print("Accuracy: %.4f (max: %.4f)" % (acc, max_acc))

            f1Sum = 0
            f1Count = 0
            for targetLabel in range(1, max(yTest)):        
                prec = getPrecision(pred_test, yTest, targetLabel)
                recall = getPrecision(yTest, pred_test, targetLabel)
                f1 = 0 if (prec+recall) == 0 else 2*prec*recall/(prec+recall)
                f1Sum += f1
                f1Count +=1    


            macroF1 = f1Sum / float(f1Count)    
            max_f1 = max(max_f1, macroF1)
            print("Non-other Macro-Averaged F1: %.4f (max: %.4f)\n" % (macroF1, max_f1))
        pred_test_ini = model.predict([sentenceTest, positionTest1, positionTest2], verbose=False)
        pred_test = predict_classes(pred_test_ini)    
        model.save('model/causal_rel_model.h5')  # creates a HDF5 file 'my_model.h5'
    def testData(folder,files):
        print("Load dataset")
        f = gzip.open(folder + 'causal-relations.pkl.gz', 'rb')
        data = pkl.load(f)
        f.close()
        maxSentenceLen = data['max_sentence_length']
        word2Idx = data['word2Idx']
        labelsMapping = data['labels_mapping']
        minDistance = data['min_distance']
        maxDistance = data['max_distance']
        print("Max Sentence Lengths: ", maxSentenceLen)
        vectorizer = Vectorizer(word2Idx, labelsMapping, minDistance, maxDistance, maxSentenceLen)
        print('files',files)
        yTest, sentenceTest, positionTest1, positionTest2 = vectorizer.vectorizeInput1(files[0])

        model = load_model('model/causal_rel_model.h5')
        pred_test_ini = model.predict([sentenceTest, positionTest1, positionTest2], verbose=False)
        pred_test = pred_test_ini.argmax(axis=-1)
        print("test result:")
        print(pred_test)
        return pred_test  
    def getModelArchitecture():
        batch_size = 64
        nb_filter = 100
        filter_length = 3
        hidden_dims = 100
        nb_epoch = 30
        position_dims = 50
        print("Load dataset")
        f = gzip.open('data/causal-relations.pkl.gz', 'rb')
        data = pkl.load(f)
        f.close()
        embeddings = data['wordEmbeddings']
        yTrain, sentenceTrain, positionTrain1, positionTrain2 = data['train_set']
        yTest, sentenceTest, positionTest1, positionTest2  = data['test_set']

        max_position = max(np.max(positionTrain1), np.max(positionTrain2))+1

        n_out = max(yTrain)+1
        #train_y_cat = np_utils.to_categorical(yTrain, n_out)
        max_sentence_len = sentenceTrain.shape[1]
        print("sentenceTrain: ", sentenceTrain.shape)
        print("positionTrain1: ", positionTrain1.shape)
        print("yTrain: ", yTrain.shape)
        print("sentenceTest: ", sentenceTest.shape)
        print("positionTest1: ", positionTest1.shape)
        print("yTest: ", yTest.shape)
        print("Embeddings: ",embeddings.shape)
        #This is a CNN for relation classification within a sentence.
        words_input = Input(shape=(max_sentence_len,), dtype='int32', name='words_input')
        words = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], trainable=False)(words_input)

        distance1_input = Input(shape=(max_sentence_len,), dtype='int32', name='distance1_input')
        distance1 = Embedding(max_position, position_dims)(distance1_input)

        distance2_input = Input(shape=(max_sentence_len,), dtype='int32', name='distance2_input')
        distance2 = Embedding(max_position, position_dims)(distance2_input)

        output = concatenate([words, distance1, distance2])

        output = Convolution1D(filters=nb_filter,
                                kernel_size=filter_length,
                                padding='same',
                                activation='tanh',
                                strides=1)(output)

        # standard max over time pooling
        output = GlobalMaxPooling1D()(output)

        output = Dropout(0.25)(output)
        output = Dense(n_out, activation='softmax')(output)
        model = Model(inputs=[words_input, distance1_input, distance2_input], outputs=[output])
        model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam', metrics=['accuracy'])
        model.summary() 
        
    def getModelArchitecture2(modelpath):
        '''showing architecture'''
        model = load_model(modelpath)
        return tf.keras.utils.plot_model(model, show_shapes=True)
        