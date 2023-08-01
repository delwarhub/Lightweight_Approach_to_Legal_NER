"""Evaluation"""




from collections import namedtuple

from collections import defaultdict

from preprocessing.tokenization import label_dataset


from prediction import predict



list_of_all_labels = ['COURT', 'PETITIONER', 'RESPONDENT', 'JUDGE', 'LAWYER', 'DATE', 'ORG', 'GPE', 'STATUTE', 'PROVISION', 'PRECEDENT', 'CASE_NUMBER', 'WITNESS', 'OTHER_PERSON']

dataset_lbld =  label_dataset(one_hot_labels=False, dev_data=True)

aaa = label_dataset(one_hot_labels=False)[1:100]

Entity = namedtuple("Entity", "e_type start_offset end_offset")

# The fuction collect_named_entities below was inspired from: https://github.com/davidsbatista/NER-Evaluation/blob/master/ner_evaluation/ner_eval.py

def collect_named_entities(tokens):
    """
    Creates a list of Entity named-tuples, storing the entity type and the start and end
    offsets of the entity.
    :param tokens: a list of tags
    :return: a list of Entity named-tuples
    """

    named_entities = []
    start_offset = None
    end_offset = None
    ent_type = None

    for offset, token_tag in enumerate(tokens):

        if token_tag == 'O':
            if ent_type is not None and start_offset is not None:
                end_offset = offset - 1
                named_entities.append(Entity(ent_type, start_offset, end_offset))
                start_offset = None
                end_offset = None
                ent_type = None

        elif ent_type is None:
            ent_type = token_tag[:-2]
            start_offset = offset

        elif ent_type != token_tag[:-2] or (ent_type == token_tag[:-2] and token_tag[-1:] == 'B'):

            end_offset = offset - 1
            named_entities.append(Entity(ent_type, start_offset, end_offset))

            # start of a new entity
            ent_type = token_tag[:-2]
            start_offset = offset
            end_offset = None

    # catches an entity that goes up until the last token

    if ent_type is not None and start_offset is not None and end_offset is None:
        named_entities.append(Entity(ent_type, start_offset, len(tokens)-1))

    return named_entities


def collect_named_entities_on_dataset(dataset, collect_true_entities = True):
    
    """
    This function iterates through each instance in the dataset and returns Entity named-tuples
    :param dataset: a tokenized dataset 
    :param collect_true_entities: if True, collect the true entities, if false collect the predicted entities
    :return: a list that contains lists (instances of the dataset) of Entity named-tuples
    """
    if collect_true_entities:
        collected_list = list()
        
        for instance in dataset:
            list_of_tokens = instance["tkn_lbls"]
            collected = collect_named_entities(list_of_tokens)
            collected_list.append(collected)
        
        return collected_list
    
    elif not collect_true_entities:
        
        collected_list = list()
        for instance in dataset:
            collected = collect_named_entities(instance)
            collected_list.append(collected)
        
        return collected_list

    
    
    
def compute_metrics_dataset(true_named, pred_named, labels, by_label = False):
    
    """
    Compute precision, recall and f1 score of entities across the dataset
    
    :param true: list of instances containing the true entities  
    :param pred: list of instances containing the predicted entities
    :param tags: list of labels
    :param by_label: if true, compute metrics for each label, if false, compute overall metrics
    :return: three dictionaries (precision, recall, f1_score) if by_label true, three floats otherwise
    """
    
        
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for index in range(len(pred_named)):
        pred_named_entities = pred_named[index]
        true_named_entities = true_named[index]
        # check for true positive and false negatives
        for pred in pred_named_entities:
            if pred in true_named_entities:
                tp[pred[0]] += 1
            elif pred not in true_named_entities:
                fn[pred[0]] += 1
        # check for false positives
        for pred in true_named_entities:
            if pred not in pred_named_entities:
                fp[pred[0]] += 1

    
    precision = dict()
    recall = dict()
    f1_score = dict()
    
    for label in labels:
        tp_label = tp[label]
        fp_label = fp[label]
        fn_label = fn[label]
        try:
            precision[label] = tp_label / (tp_label + fp_label)
            recall[label] = tp_label / (tp_label + fn_label)
            f1_score[label] = (2 * precision[label] * recall[label]) / (precision[label] + recall[label])
        except Exception:
            precision[label] = "N/A"
            recall[label] = "N/A"
            f1_score[label] = "N/A"
    
    if by_label:
        
        return precision, recall, f1_score
    
    elif not by_label:
        
        tp_overall = sum(tp.values())
        fp_overall = sum(fp.values())
        fn_overall = sum(fn.values())
        
        try:
            precision_overall = tp_overall / (tp_overall + fp_overall)
            recall_overall = tp_overall / (tp_overall + fn_overall)            
            f1_score_overall = (2 * precision_overall * recall_overall) / (precision_overall + recall_overall)
        except Exception:
            precision_overall = "N/A"
            recall_overall = "N/A"
            f1_score_overall = "N/A"

            
        return precision_overall, recall_overall, f1_score_overall

def compute_metrics_dataset_partial(true_named, pred_named, labels, by_label = False):

    """
    Compute Type match precision, recall and f1 score of entities across the dataset
    
    :param true: list of instances containg the true entities  
    :param pred: list of instances containg the predicted entities
    :param tags: list of labels
    :param by_label: if true, compute metrics for each label, if false, compute overall metrics
    :return: three dictionaries (Type match precision, recall, f1_score) if by_label true, three floats otherwise
    """

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    # iterate through the instances 
    for index in range(len(true_named)):
        true_text = true_named[index]
        pred_text = pred_named[index]
        
        # store true positives to detect false positive and false negatives later
        tp_in_pred = []
        tp_in_true = []
        
        # check for true positive
        for pred_entity in pred_text:
            for true_entity in true_text:
                # entity labels is the same, and pred_entity partially overlaps with true_entity
                if pred_entity[0] == true_entity[0] and pred_entity[1] >= true_entity[1] and not pred_entity[1] > true_entity[2] \
                or pred_entity[0] == true_entity[0] and pred_entity[2] <= true_entity[2] and not pred_entity[2] < true_entity[1]:

                    tp[pred_entity[0]] += 1
                    tp_in_pred.append(pred_entity)
                    tp_in_true.append(true_entity)
                    
        # check for false negatives
        for pred_entity in pred_text:
            if pred_entity not in tp_in_pred:
                fp[pred_entity[0]] += 1
                
        # check for false negatives        
        for true_entity in true_text:
            if true_entity not in tp_in_true:
                fn[true_entity[0]] += 1

    precision = dict()
    recall = dict()
    f1_score = dict()
    
    for label in labels:
        tp_label = tp[label]
        fp_label = fp[label]
        fn_label = fn[label]
        
        try:
            precision[label] = tp_label / (tp_label + fp_label)
            recall[label] = tp_label / (tp_label + fn_label)
            f1_score[label] = (2 * precision[label] * recall[label]) / (precision[label] + recall[label])
        except Exception:
            precision[label] = "N/A"
            recall[label] = "N/A"
            f1_score[label] = "N/A"
    
    if by_label:
        
        return precision, recall, f1_score
    
    elif not by_label:
        
        tp_overall = sum(tp.values())
        fp_overall = sum(fp.values())
        fn_overall = sum(fn.values())
        
        try:
            precision_overall = tp_overall / (tp_overall + fp_overall)
            recall_overall = tp_overall / (tp_overall + fn_overall)            
            f1_score_overall = (2 * precision_overall * recall_overall) / (precision_overall + recall_overall)
        except Exception:
            precision_overall = "N/A"
            recall_overall = "N/A"
            f1_score_overall = "N/A"

            
        return precision_overall, recall_overall, f1_score_overall


if __name__  == "__main__":  
    

  # dataset_lbld is a dictionary that contains the instances from the dev dataset
  # dataset_small = dataset_lbld[1:50]

  dataset_small = dataset_lbld

  #print(dataset_small)
  
  # this function takes in a section of instances from the dev dataset and outputs
  # change 2nd and 3rd arguments to the parameters you prefer
  # prediction_on_dataset_small = predict(dataset_small, "conv_deep", "deepConvHead_3batch.pt")

  # prediction_on_dataset_small = predict(dataset_small, "conv_shallow", "../model/shallowConvHead_3batch.pt")
  # prediction_on_dataset_small = predict(dataset_small, "conv_deep", "../model/deepConvHead_3batch.pt")
  prediction_on_dataset_small = predict(dataset_small, "linear", "../model/linearHead_3batch.pt")
  # prediction_on_dataset_small = predict(dataset_small, "linear", "../model/linearDropoutHead_3batch.pt")


  # print(prediction_on_dataset_small)
  
  # true_named and pre_named both are lists of instances. Inside each instance list there are entity named tuples
  
  # collect_named_entities_on_dataset takes in a list of tokenized sentences and gives back a list of Entity tuples
  true_named = collect_named_entities_on_dataset(dataset_small, collect_true_entities=True)
  # print(true_named)
  
  pred_named = collect_named_entities_on_dataset(prediction_on_dataset_small, collect_true_entities=False)
  # print(pred_named)

  
  # true_named and pre_named are then compared using compute_metrics_dataset
  # print(compute_metrics_dataset(true_named, pred_named, list_of_all_labels, by_label = True))
  # print(compute_metrics_dataset(true_named, pred_named, list_of_all_labels, by_label = False))

  print(compute_metrics_dataset_partial(true_named, pred_named, list_of_all_labels, by_label=False))

  """
  # Test compute_metrics_dataset_partial()
  
    true_experiment_partial = [
                      [Entity(e_type='RESPONDENT', start_offset=9, end_offset=17), Entity(e_type='PETITIONER', start_offset=52, end_offset=56),
                      Entity(e_type='RESPONDENT', start_offset=66, end_offset=71)],
                     
                     [Entity(e_type='COURT', start_offset=4, end_offset=17), Entity(e_type='PETITIONER', start_offset=3, end_offset=8), Entity(e_type='PETITIONER', start_offset=20, end_offset=30),  Entity(e_type='RESPONDENT', start_offset=234, end_offset=236)],
                     
                      [Entity(e_type='RESPONDENT', start_offset=265, end_offset=278), Entity(e_type='RESPONDENT', start_offset=290, end_offset=305), Entity(e_type='ONLY IN TRUE', start_offset=260, end_offset=289)]
                      ]
  
    pred_experiment_partial = [
                      [Entity(e_type='RESPONDENT', start_offset=1, end_offset=6), Entity(e_type='PETITIONER', start_offset=52, end_offset=60),
                      Entity(e_type='RESPONDENT', start_offset=66, end_offset=71)],
                     
                     [Entity(e_type='COURT', start_offset=2, end_offset=15), Entity(e_type='PETITIONER', start_offset=9, end_offset=21), Entity(e_type='PETITIONER', start_offset=12, end_offset=30),
                      Entity(e_type='RESPONDENT', start_offset=234, end_offset=236)],
                     
                      [Entity(e_type='RESPONDENT', start_offset=279, end_offset=280), Entity(e_type='RESPONDENT', start_offset=260, end_offset=289),Entity(e_type='ONLY IN PRED', start_offset=260, end_offset=289) ]
                      ]
    list_of_labels_experiment_partial = ['COURT', 'RESPONDENT', 'PETITIONER', 'JUDGE']
  
  
    print(compute_metrics_dataset_partial(true_experiment_partial, pred_experiment_partial, list_of_labels_experiment_partial, by_label = True))
  """
