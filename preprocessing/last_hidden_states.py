"""Feed the instances through a RobertaModel transformer and get the last hidden states"""
from tokenization import label_dataset
from transformers import RobertaModel
import torch
import h5py
from tqdm import tqdm

# get torch device:
pt_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = RobertaModel.from_pretrained("roberta-base").to(pt_device)


def get_last_hidden_states(input_ids: bool = True, attention_mask: bool = True, tkn_lbls: bool = True) -> list:
    """
    Get the transformer's last hidden states of all instances in the dataset
    
    :param input_ids: If False, input_ids is omitted from the dictionary
    :param attention_mask: If False, attention_mask is omitted from the dictionary
    :param tkn_lbls: If False, tkn_lbls is omitted from the dictionary
    :return: List of dicts with labeled token instances and last hidden states
    """
    data = label_dataset()

    lbl_dataset_hidden = list()

    for instance in data:
        # pass the instance through the model and get last hidden state
        output = model(instance["input_ids"], instance["attention_mask"])
        last_hidden_states = output.last_hidden_state
        
        instance['last_hidden_states'] = last_hidden_states
        
        # include or not optional keys in the output dictionary
        if input_ids == False:
            del instance["input_ids"]
        if attention_mask == False:
            del instance["attention_mask"]
        if tkn_lbls == False:
            del instance["tkn_lbls"]

        lbl_dataset_hidden.append(instance)
    return lbl_dataset_hidden


def store_last_hidden_states(file_path: str, dev_data: bool = False) -> None:
    """
    Get the transformer's last hidden states of all instances in the dataset.
    
    :param file_path: Path for the HDF5 file containing last hidden states and token labels.
    :param dev_data: If True, the development data is processed.
    :return: None (create HDF5 file).
    """
    data = label_dataset(dev_data=dev_data)

    # create HDF5 file
    f = h5py.File(file_path, 'w')
    
    # for each instance create a HDF5 group. Inside each group store last hidden states and token labels 
    for instance in tqdm(data):
        with torch.no_grad():
            # pass the instance through the model and get last hidden state
            output = model(instance["input_ids"], instance["attention_mask"])
            last_hidden_states = output.last_hidden_state
            
            instance['last_hidden_states'] = last_hidden_states
            
            # create a group to store all data of an instance
            instance_grp = f.create_group(instance["id"])
            
            # convert into numpy array to store them into HDF5 file
            labels_array = (instance["tkn_lbls"]).detach().numpy()
            hidden_array = (instance["last_hidden_states"]).detach().numpy()

            # store the two dataset into the instance group 
            instance_grp.create_dataset("tkn_lbls", data = labels_array)
            instance_grp.create_dataset("last_hidden_states", data = hidden_array)

    f.close()

    
if __name__ == "__main__":
    # print(get_last_hidden_states())
    # print(get_last_hidden_states(attention_mask = False,  tkn_lbls = False))
    # print(get_last_hidden_states(input_ids = False))
    # print(get_last_hidden_states(input_ids = False, attention_mask = False,  tkn_lbls = False))

    # generate and store training dataset file:
    store_last_hidden_states("./data/roberta_inference_full_train.hdf5")

    # generate and store development dataset file:
    # store_last_hidden_states("./data/roberta_inference_full_dev.hdf5", dev_data=True)

    """
    test_data = h5py.File('roberta_inference.hdf5', 'r')

    # print(list(test_data.keys()))
    # print(test_data.keys())
    # print(test_data.items())
    # print(list(test_data.items()))

    for inst_id in list(test_data.keys()):
        print(inst_id)
        # print(test_data[inst_id])
        # print(test_data[inst_id].items())
        # print(test_data[inst_id].keys())

        # print(list(test_data[inst_id].keys()))

        # print(test_data[inst_id].values())
        # print(list(test_data[inst_id].values()))

        # print(test_data[inst_id]['last_hidden_states'])

        # print(test_data[inst_id]['last_hidden_states'][()])

        test_array = test_data[inst_id]['last_hidden_states']
        print(test_array)
        # print(test_array[()])

        test_array_1 = test_array[()]
        print(test_array_1)
        # print(test_array_1.dtype)

        # print(test_array.read_direct())
        # print(test_array.read_direct(test_array))

        # print(test_array.astype('int16'))

        # print(test_array.astype('float64')[:])

        break

    # print(test_data[''])
    """