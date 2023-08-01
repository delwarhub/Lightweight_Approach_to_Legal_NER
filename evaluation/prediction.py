"""
This file contains two function:
The first one store the prediction of the model into a HDF5 file
The second one extract the true labels from the tokenized dataset
"""

"""

IMPORTANT: infer_text_labels inside roberta_legal_ner needs to return the prediction,
           not print it

"""

import h5py
import h5py
from data.data_util import get_instance_by_id
from preprocessing.tokenization import label_dataset
from model.roberta_legal_ner import infer_text_labels
from data.data_util import get_instance_by_id_dev


dataset_lbld =  label_dataset(one_hot_labels=False, dev_data=True)

# Store all predictions into HDF5 file
def store_h5py(dataset, file_path) -> None:
    """
    Store label predictions on HDF5 file. 
    Each instance in dataset is contained into HDF5 dataset that has as key the instance ID and that contains the instance predicted labels 
    
    :param dataset: the labeled dataset (a dict) obtained using the label_dataset function
    :param file_path: path of the file generated with this function
    :return: None (create HDF5 file)
    """
    with h5py.File(file_path, 'w') as f:
        for instance in dataset:
            id_num = instance["id"]
            instance_dataset = get_instance_by_id(id_num)
            text = instance_dataset["data"]["text"]
            pred_tokens = infer_text_labels(text, "linear", "linearDropoutHead_3batch.pt")
            #pred_tokens_arr = np.array(pred_tokens)
            #pred_tokens_dtype = h5py.string_dtype(encoding='utf-8')
            f.create_dataset(id_num, data = pred_tokens)
        print("All predicted tokens had been stored")

# get the true labels 
def get_true_labels(dataset) -> list:
    """
    Return a list of true labels from the data
    
    :param dataset: the labeled dataset (a dict) obtained using the label_dataset function
    :return: list of true labels
    """

    tkn_lbl_true = list()
    for i in dataset:
        tkn_lbl = i["tkn_lbls"]
        tkn_lbl_true.append(tkn_lbl)
    
    return tkn_lbl_true

def predict(dataset, head_class: str, head_path: str):
    
    predictions = list()
    for instance in dataset:
    
        id_num = instance["id"]
        instance_dataset = get_instance_by_id_dev(id_num)
        text = instance_dataset["data"]["text"]
        pred_tokens = infer_text_labels(text, head_class, head_path)
        
        predictions.append(pred_tokens)
    return predictions
        
    

if __name__  == "__main__":
    
    # store_h5py(dataset_lbld[9000:9004], "store_a_file.hdf5")
    # print(get_true_labels(dataset_lbld[9000:9004]))
    
    #small_dataset = dataset_lbld[9000:9004]
    
    dataset_small = dataset_lbld[1:10]
    
    print(dataset_small)
    
    print(predict(dataset_small))
    
    
    
