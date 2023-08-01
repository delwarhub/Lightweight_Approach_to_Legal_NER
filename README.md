# Lightweight approach to Legal NER 
Repository for the final group project as part of the Advanced Natural Language Processing course at Universität Potsdam
in the winter semester 2022/23.  
By Emanuele DeRossi, Md.Delwar Hossain and Jonathan Jordan

## General
Project based on theoretical aspects and data from https://github.com/Legal-NLP-EkStep/legal_NER .  
Code is the work of the authors unless specified otherwise here or in code comments.  
https://github.com/huggingface/transformers/blob/v4.27.0/src/transformers/models/roberta/modeling_roberta.py#L1360 used 
as reference for the RoBERTaLegalNER model class in model/roberta_legal_ner.py.  
The function collect_named_entities() in evaluation/evaluation.py was inspired by 
https://github.com/davidsbatista/NER-Evaluation/blob/master/ner_evaluation/ner_eval.py

## Usage
### Source dataset acquisition
Run data/data_util.py to retrieve and unpack the source dataset JSON files. This is required for all further use.
### Source dataset exploration
Run preprocessing/exploration.py to examine a few aspects of the source dataset, like dataset instances with a specified 
amount of labeled spans, the actual format of an instance in the source dataset JSON files, text lengths of the two 
different judgement and preamble parts, and adjacent labeled spans.
### Tokenization and token labeling
Run preprocessing/tokenization.py to assess dataset instances that will be discarded due to containing too many RoBERTa 
BPE tokens, to show the result of preprocessing a single dataset instance and to assess the need for BIO token labels 
via adjacency of labeled token spans in the preprocessed training dataset.
### Training dataset preprocessing and storage
Run preprocessing/last_hidden_states.py to preprocess the training dataset and save it as a HDF5 file to be used for 
training. This is necessary for model training using model/linear_heads.py and model/convolution_heads.py.
### Head model training
Run model/linear_heads.py to perform a sample training run of a SimpleLinearLNERHead model class instance with dropout.  
Run model/convolution_heads.py to perform a sample training run of a ConvolutionLNERHeadDeep model class instance.
### Full model inference
Run model/roberta_legal_ner.py to use a RobertaLegalNER model class instance with a SimpleLinearLNERHead classifier 
head model trained without dropout for 70 epochs to infer token labels for an example sentence from the training 
dataset.  
This demonstrates the core task the head models are trained for.
### Evaluation
Run evaluation/evaluation.py to evaluate a RobertaLegalNER model class instance with a SimpleLinearLNERHead classifier 
head model trained without dropout for 70 epochs on the development dataset, using the partial/type match criterion.