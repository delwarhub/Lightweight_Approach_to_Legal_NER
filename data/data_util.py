"""Utility module for data loading and extraction."""

import json
from typing import Union

from zipfile import ZipFile
from io import BytesIO
from urllib.request import urlopen

# dataset URLs:
TRAIN_URL = "https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/NER/NER_TRAIN.zip"
DEV_URL = "https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/NER/NER_DEV.zip"


def download_and_extract(url_train: str, url_dev: str, target_dir: str) -> None:
    """
    :param url_train: the url of the training dataset (TRAIN_URL)
    :param url_dev: the url of the development dataset (DEV_URL)
    :param target_dir: the folder where the data will be stored
    """
    
    # download and extract train data
    with urlopen(url_train) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall(target_dir)
    print("Train data extracted!")

    # download and extract dev data
    with urlopen(url_dev) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall(target_dir)
    print("Dev data extracted!")


TRAIN_JUD_PATH = "../data/NER_TRAIN_JUDGEMENT.json"
TRAIN_PRE_PATH = "../data/NER_TRAIN_PREAMBLE.json"

DEV_JUD_PATH = "../data/NER_DEV_JUDGEMENT.json"
DEV_PRE_PATH = "../data/NER_DEV_PREAMBLE.json"


def load_data_json(file_path: str) -> list:
    """
    Loads and parses JSON training data.

    :param file_path: String containing the file path of the judgement data.
    :return: List of data instances.
    """
    return json.loads(open(file_path).read())


def load_train_part(part: str = '', verbose: bool = False) -> list:
    """
    Loads either part or all of the training data.

    :param part: String specifying part; 'pre'=preambles, 'jdg'=judgements; loads pre+jdg if omitted.
    :param verbose: Print loaded part notice.
    :return: List of data instances.
    """
    if part == 'jdg':
        if verbose:
            print("Loaded JUDGEMENT training data.")
        return load_data_json(TRAIN_JUD_PATH)
    elif part == 'pre':
        if verbose:
            print("Loaded PREAMBLE training data.")
        return load_data_json(TRAIN_PRE_PATH)
    else:
        if verbose:
            print("Loaded PREAMBLE and JUDGEMENT training data.")
        return load_data_json(TRAIN_PRE_PATH) + load_data_json(TRAIN_JUD_PATH)


def load_dev_part(part: str = '', verbose: bool = False) -> list:
    """
    Loads either part or all of the development data.

    :param part: String specifying part; 'pre'=preambles, 'jdg'=judgements; loads pre+jdg if omitted.
    :param verbose: Print loaded part notice.
    :return: List of data instances.
    """
    if part == 'jdg':
        if verbose:
            print("Loaded JUDGEMENT dev data.")
        return load_data_json(DEV_JUD_PATH)
    elif part == 'pre':
        if verbose:
            print("Loaded PREAMBLE dev data.")
        return load_data_json(DEV_PRE_PATH)
    else:
        if verbose:
            print("Loaded PREAMBLE and JUDGEMENT dev data.")
        return load_data_json(DEV_PRE_PATH) + load_data_json(DEV_JUD_PATH)


def get_instance_by_id(instance_id: str) -> dict:
    """
    Get a dataset instance from train data by its dataset ID.

    :param instance_id: Dataset instance ID of the instance to get.
    :return: Dataset instance dict.
    """
    data = load_train_part()

    for instance in data:
        if instance['id'] == instance_id:
            return instance


def get_instance_by_id_dev(instance_id: str) -> dict:
    """
    Get a dataset instance from dev data by its dataset ID.

    :param instance_id: Dataset instance ID of the instance to get.
    :return: Dataset instance dict.
    """
    data = load_dev_part()

    for instance in data:
        if instance['id'] == instance_id:
            return instance


# Label types for mapping:
# following order from source dataset paper, including BIO tags
LABEL_TYPES = [
    'O',  # outside BIO label
    'COURT-B', 'COURT-I',
    'PETITIONER-B', 'PETITIONER-I',
    'RESPONDENT-B', 'RESPONDENT-I',
    'JUDGE-B', 'JUDGE-I',
    'LAWYER-B', 'LAWYER-I',
    'DATE-B', 'DATE-I',
    'ORG-B', 'ORG-I',
    'GPE-B', 'GPE-I',
    'STATUTE-B', 'STATUTE-I',
    'PROVISION-B', 'PROVISION-I',
    'PRECEDENT-B', 'PRECEDENT-I',
    'CASE_NUMBER-B', 'CASE_NUMBER-I',
    'WITNESS-B', 'WITNESS-I',
    'OTHER_PERSON-B', 'OTHER_PERSON-I'
]

label_to_lblid_dict = {label: LABEL_TYPES.index(label) for label in LABEL_TYPES}
lblid_to_label_dict = {LABEL_TYPES.index(label): label for label in LABEL_TYPES}


def convert_lblid(label: Union[str, int]) -> Union[str, int]:
    """
    Convert label type strings to integer label ids and vice versa.

    :param label: Label type strings to integer label id.
    :return: Label type strings to integer label id.
    """
    if type(label) == int:
        return lblid_to_label_dict[label]
    elif type(label) == str:
        return label_to_lblid_dict[label]


if __name__ == "__main__":
    # download and extract the JSON source datasets:
    download_and_extract(TRAIN_URL, DEV_URL, ".")

    # print(load_train_part())

    """
    print(label_to_lblid_dict)
    print(lblid_to_label_dict)

    print("Label ID 1:", convert_lblid(1))
    print("RESPONDENT-I ID:", convert_lblid('RESPONDENT-I'))

    print("RESPNDENT-I ID:", convert_lblid('RESPNDENT-I'))
    """
