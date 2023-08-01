"""Tokenization"""
import json
from typing import Union

from data.data_util import load_train_part, load_dev_part, get_instance_by_id, convert_lblid, LABEL_TYPES
from transformers import RobertaTokenizerFast
import torch

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")


def check_token_match(part: str = '', minimise: bool = True) -> list:
    """
    Check token-label match.

    :param part: String to select dataset part; 'jdg' or 'pre', omit to check whole dataset.
    :param minimise: If true, token-label coverage is minimized (only tokens fully inside labeled span; 'if tokens in
    span'), otherwise it is maximized (tokens can extend beyond labeled span; 'if span in tokens').
    :return: List of mismatches between character label spans and token label spans.
    """
    data = load_train_part(part)

    mismatch_list = []
    for inst in range(len(data)):
        inst_txt = data[inst]['data']['text']
        inst_tkn = tokenizer(inst_txt, return_offsets_mapping=True)

        inst_mismatch_list = []

        spans = []
        for annotation in data[inst]['annotations'][0]['result']:
            start = annotation['value']['start']
            end = annotation['value']['end']
            label = annotation['value']['labels'][0]
            lbl_span = (start, label, end)
            spans.append(lbl_span)

        for span in spans:
            span_tkn_spans = []
            for tkn_offset_i in range(1, len(inst_tkn['offset_mapping'])-1):
                if minimise:
                    if inst_tkn['offset_mapping'][tkn_offset_i][0] >= span[0] and inst_tkn['offset_mapping'][tkn_offset_i][1] <= span[2]:
                        span_tkn_spans.append(inst_tkn['offset_mapping'][tkn_offset_i])
                        start_mismatch = span_tkn_spans[0][0] - span[0]
                        end_mismatch = span_tkn_spans[-1][1] - span[2]
                else:
                    if inst_tkn['offset_mapping'][tkn_offset_i][1] >= span[0] and inst_tkn['offset_mapping'][tkn_offset_i][0] <= span[2]:
                        span_tkn_spans.append(inst_tkn['offset_mapping'][tkn_offset_i])
                        start_mismatch = span_tkn_spans[0][0] - span[0]
                        end_mismatch = span_tkn_spans[-1][1] - span[2]

            mismatch = (start_mismatch, end_mismatch, len(span_tkn_spans))
            inst_mismatch_list.append(mismatch)

        mismatch_list.append(inst_mismatch_list)

    return mismatch_list


def check_tokenized_length(dataset: list, threshold: int = 512) -> list:
    """
    Checks the length of tokenized data instances, returning IDs of those that are too long for the RoBERTa model.

    :param dataset: Dataset list of instance dicts.
    :param threshold: Int threshold for number of tokens.
    :return: List of (instance ID, number of tokens) tuples.
    """
    inst_id_len_list = []
    for inst in range(len(dataset)):
        inst_id = dataset[inst]['id']
        inst_txt = dataset[inst]['data']['text']
        inst_tkn = tokenizer(inst_txt)
        if len(inst_tkn['input_ids']) > threshold:
            inst_id_len_list.append((inst_id, len(inst_tkn['input_ids'])))

    return inst_id_len_list


def labels_to_one_hot(label_list: list) -> list:
    """
    Convert string label list to list of one-hot lists.

    :param label_list: List of string labels.
    :return: List of one-hot token label lists.
    """
    # return [[1 if convert_lblid(label) == i else 0 for i in range(len(LABEL_TYPES)-1)] for label in label_list]
    return [[1 if convert_lblid(label) == i else 0 for i in range(len(LABEL_TYPES))] for label in label_list]


def label_tokens(instance: dict, tkn_threshold: int = 512, one_hot_labels: bool = True) -> Union[dict, None]:
    """
    Tokenize data instance and assign labels to tokens. Uses maximum overlap: If a token is partially in a labeled
    span, it is assigned the label of the span. Applies labels with BIO tags to prevent issues with adjacent spans
    with the same label.
    Checks for token amount and returns None if instance has too many tokens.
    Labels are converted into one-hot representation by default.

    :param instance: A data instance from the dataset.
    :param tkn_threshold: Int threshold for number of tokens.
    :param one_hot_labels: If True, token labels are converted to one-hot list.
    :return: Dict with instance ID, list of tokens, attention mask and list of labels.
    """
    inst_txt = instance['data']['text']
    # tokenize text and get token offsets:
    inst_tkn = tokenizer(inst_txt, return_offsets_mapping=True)

    # if too many tokens, return None to discard:
    if len(inst_tkn['input_ids']) > tkn_threshold:
        return None

    # create dict holding instance id, instance tokens and attention mask:
    inst_lbl_tokens = dict()
    inst_lbl_tokens['id'] = instance['id']
    inst_lbl_tokens['input_ids'] = inst_tkn['input_ids']
    inst_lbl_tokens['attention_mask'] = inst_tkn['attention_mask']

    # get labeled spans:
    lbl_spans = []
    for annotation in instance['annotations'][0]['result']:
        start = annotation['value']['start']
        end = annotation['value']['end']
        label = annotation['value']['labels'][0]
        lbl_span = (start, label, end)
        lbl_spans.append(lbl_span)

    # create list of token labels, initialized to 'outside' labels for all instance tokens:
    tkn_lbls = ['O' for i in range(len(inst_tkn['input_ids']))]

    # get partial overlap:
    # for all labeled spans:
    for lbl_span in lbl_spans:
        # make sure that each beginning of a labeled span is properly labeled:
        begin_tagged = False
        # for all relevant instance tokens, ignoring the sequence start and end tokens:
        for tkn_offset_i in range(1, len(inst_tkn['offset_mapping']) - 1):
            cur_tkn_offset = inst_tkn['offset_mapping'][tkn_offset_i]
            # if a token has at least partial overlap with a labeled span:
            if cur_tkn_offset[0] >= lbl_span[0] and cur_tkn_offset[1] <= lbl_span[2]:
                # if it is the first token to be labeled in a span, assign 'beginning' label:
                if not begin_tagged:
                    tkn_lbls[tkn_offset_i + 1] = f"{lbl_span[1]}-B"
                    begin_tagged = True
                    # otherwise assign 'inside' label:
                else:
                    tkn_lbls[tkn_offset_i + 1] = f"{lbl_span[1]}-I"

    if one_hot_labels:
        # convert readable token labels to one-hot representation:
        inst_lbl_tokens['tkn_lbls'] = labels_to_one_hot(tkn_lbls)
    else:
        inst_lbl_tokens['tkn_lbls'] = tkn_lbls

    return inst_lbl_tokens


def lbl_tkn_inst_tensors(tkn_lbl_instance: dict) -> dict:
    """
    Turns the token and one-hot label data of a labeled token instance into tensors.

    :param tkn_lbl_instance: Labeled token instance dict.
    :return: Labeled token instance dict with token data as tensors.
    """
    tensor_input_ids = torch.unsqueeze(torch.LongTensor(tkn_lbl_instance['input_ids']), 0)
    tensor_attention_mask = torch.unsqueeze(torch.LongTensor(tkn_lbl_instance['attention_mask']), 0)
    # only convert one-hot labels to tensor:
    if type(tkn_lbl_instance['tkn_lbls'][0]) == list:
        tensor_one_hot_labels = torch.LongTensor(tkn_lbl_instance['tkn_lbls'])
    else:
        tensor_one_hot_labels = tkn_lbl_instance['tkn_lbls']

    return {'id': tkn_lbl_instance['id'], 'input_ids': tensor_input_ids, 'attention_mask': tensor_attention_mask,
            'tkn_lbls': tensor_one_hot_labels}


def label_dataset(convert_to_tensors: bool = True, one_hot_labels: bool = True, dev_data: bool = False) -> list:
    """
    Tokenize and label dataset.

    :param convert_to_tensors: If True, input_ids, attention_mask and one-hot labels are converted to torch tensors.
    :param one_hot_labels: If True, token labels are converted to one-hot list.
    :param dev_data: If True, the development dataset is labeled, else training dataset.
    :return: List of all tokenized and labeled instances.
    """
    if dev_data:
        data = load_dev_part()
    else:
        data = load_train_part()

    tkn_lbl_instances = []
    for instance in data:
        tkn_lbl_instance = label_tokens(instance, one_hot_labels=one_hot_labels)
        # if the instance has too many tokens, None is returned by label_tokens()
        if tkn_lbl_instance:
            if convert_to_tensors:
                tkn_lbl_instances.append(lbl_tkn_inst_tensors(tkn_lbl_instance))
            else:
                tkn_lbl_instances.append(tkn_lbl_instance)

    return tkn_lbl_instances


def export_tkn_lbl_dataset_json(file_name: str = 'readable_tkn_lbl_dataset.json') -> None:
    """
    Exports easily readable tokenized and labeled dataset as JSON file.

    :param file_name: Name of the JSON file to export.
    :return: None, file output.
    """
    with open(file_name, 'w') as out_file:
        out_file.write(json.dumps(label_dataset(convert_to_tensors=False, one_hot_labels=False)))


def check_lbl_tkn_adjacency(use_file: str = '', return_instance_list: bool = False) -> Union[None, list]:
    """
    Check the labeled token dataset for labeled token span adjacency, ie BIO-tags B and directly preceding I. Print
    count of adjacent labeled token spans.

    :param use_file: Name of JSON file containing readable labeled token dataset to be loaded; if empty string
    label_dataset() will be used to get data.
    :param return_instance_list: If True, return list of instance-IDs and adjacent labeled tokens of different spans.
    :return: Print output.
    """
    if use_file:
        with open(use_file, 'r') as in_file:
            lbl_tkn_data = json.load(in_file)
    else:
        lbl_tkn_data = label_dataset(convert_to_tensors=False, one_hot_labels=False)

    adjacent_count = 0
    if return_instance_list:
        adjacent_list = []
    for instance in lbl_tkn_data:
        tkn_lbls = instance['tkn_lbls']
        for tkn_lbl_i in range(len(tkn_lbls)):
            if tkn_lbls[tkn_lbl_i][-1] == "B":
                if tkn_lbl_i - 1 >= 0:
                    if not tkn_lbls[tkn_lbl_i - 1] == "O":
                        adjacent_count += 1
                        if return_instance_list:
                            adjacent_list.append((instance['id'], tkn_lbls[tkn_lbl_i - 1], tkn_lbls[tkn_lbl_i]))
    print(f"Number of adjacent labeled token spans: {adjacent_count}")
    if return_instance_list:
        return adjacent_list


if __name__ == "__main__":

    data_jdg = load_train_part('jdg')
    jdg_too_long = len(check_tokenized_length(data_jdg))
    jdg_len = len(data_jdg)
    print(f"Judgement data instances above 512 tokens: {jdg_too_long} of {jdg_len} ({(jdg_too_long/jdg_len)*100}%)")

    data_pre = load_train_part('pre')
    pre_too_long = len(check_tokenized_length(data_pre))
    pre_len = len(data_pre)
    print(f"Preamble data instances above 512 tokens: {pre_too_long} of {pre_len} ({(pre_too_long/pre_len)*100}%)")

    print("-----\nTesting token labeling...")
    test_instance = get_instance_by_id('00212ec9f8a345d3a12c1022e2deab1e')
    print("\nTest instance data:")
    print(test_instance)
    print("\nLabeled test instance - readable:")
    print(label_tokens(test_instance, one_hot_labels=False))
    print("\nLabeled test instance - processable; ie one-hot labels, torch LongTensors:")
    print(lbl_tkn_inst_tensors(label_tokens(test_instance)))

    # check one-hot label conversion:
    # print(labels_to_one_hot(['O', 'O', 'PRECEDENT-B']))

    # check processing of whole dataset:
    # print(label_dataset())

    # check export of readable dataset:
    export_tkn_lbl_dataset_json()

    print("-----\nChecking labeled token span adjacency...")
    check_lbl_tkn_adjacency(use_file='readable_tkn_lbl_dataset.json')
    # print(check_lbl_tkn_adjacency(use_file='readable_tkn_lbl_dataset.json', return_instance_list=True))
    print("Due to the amount of adjacent labeled token spans, BIO tagging is necessary.")

    """
    a = check_token_match('jdg', minimise=True)
    b = check_token_match('jdg', minimise=False)
    for i in range(len(a)):
        print(a[i], "\n", b[i], "\n")
    """
