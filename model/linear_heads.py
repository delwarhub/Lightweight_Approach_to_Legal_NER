"""Linear NN classifier heads for legal NER using RoBERTa-base last hidden states"""

from typing import Union, Optional

import torch
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss

import h5py
from h5py import File as h5pyFile

from tqdm import tqdm

import random
from statistics import mean

from data.data_util import LABEL_TYPES


# number of label classes:
num_labels = len(LABEL_TYPES)

# roberta inference hidden state size:
hidden_size = 768

# get torch device:
pt_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SimpleLinearLNERHead(nn.Module):
    """
    Simple linear legal-NER head with dropout using RoBERTa inference outputs.
    This head is based on the huggingface RobertaForTokenClassification architecture, but as a separate model
    head/layer for finetune training. RoBERTa part of the intended full model is assumed to remain frozen, whereas for
    the huggingface RobertaForTokenClassification model it was trained as well.
    https://huggingface.co/docs/transformers/model_doc/roberta#transformers.RobertaForTokenClassification
    Use drop_p=0.0 to have this model class behave like a simple linear head without dropout.
    """
    def __init__(self, drop_p: float = 0.5):
        super().__init__()
        self.dropout = nn.Dropout(drop_p)
        self.linear = nn.Linear(hidden_size, num_labels)

    def forward(self, roberta_hidden_states: Tensor) -> Tensor:
        drop_output = self.dropout(roberta_hidden_states)
        logits = self.linear(drop_output)

        return logits


def simple_linear_head_train_loop(model: SimpleLinearLNERHead, data: h5pyFile,
                                  save_path: str = '', save_name: str = 'simpleLinearLNERHead',
                                  epochs_per_ckpt: int = 0, save_final_epoch_ckpt: bool = False,
                                  rnd_instance_order: bool = True, rnd_instance_seed: Union[None, int] = None,
                                  batch_size: int = 1, drop_p: float = 0.0,
                                  lr: float = 0.001, epochs: int = 3, initial_epoch: int = 0,
                                  ckpt_optimizer: Union[None, torch.optim.Adam] = None):
    """
    Train a SimpleLinearLNERHead model. Optionally save trained model and checkpoints
    (including training parameters).

    :param model: SimpleLinearLNERHead model instance.
    :param data: h5py File instance.
    :param save_path: Path of directory to save trained model file at. If omitted, model will not be saved! Pass . to
    save in current directory.
    :param save_name: Name for the saved model file(s).
    :param epochs_per_ckpt: Save a checkpoint every epochs_per_ckpt epochs. If 0, no checkpoints will be saved.
    :param save_final_epoch_ckpt: Save checkpoint after last epoch of training run.
    :param rnd_instance_order: Randomize order of training data instances each epoch. Uses random python std lib. Uses
    distinct random.Random class instance RNG to prevent interference with other randomized processes.
    :param rnd_instance_seed: Seed for the training data instance randomization RNG; for reproducibility. If None
    (default) system time is used.
    :param batch_size: Size of training batches for batch GD. Gradients are applied for each batch. 1 = SGD
    :param drop_p: Dropout probability; for checkpoint parameter preservation.
    :param lr: Learning rate.
    :param epochs: Number of epochs.
    :param initial_epoch: Epoch number to start with; for continued training from saved checkpoint.
    :param ckpt_optimizer: Use optimizer with states from saved checkpoint.
    :return: None, model trained in-place.
    """
    # Cross Entropy loss function:
    loss_fct = CrossEntropyLoss()

    # use checkpoint optimizer if given:
    if ckpt_optimizer:
        optimizer = ckpt_optimizer
    else:
        # basic Adam optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # initial instance key list:
    inst_key_list = list(data.keys())

    # number of batches:
    n_batches = int(len(inst_key_list) / batch_size)
    print("Number of batches:", n_batches)

    # instance order randomization RNG:
    if rnd_instance_order:
        inst_order_rng = random.Random(rnd_instance_seed)

    train_run_epochs = initial_epoch + epochs

    # cur_epoch = 0
    cur_epoch = initial_epoch
    while cur_epoch < train_run_epochs:
        print(f"Training epoch {cur_epoch}")

        # instance order randomization:
        if rnd_instance_order:
            inst_order_rng.shuffle(inst_key_list)

        # loss accumulation list:
        loss_list = []

        for batch_n in tqdm(range(n_batches)):
            # reset gradients for each batch:
            optimizer.zero_grad()
            # get instance keys ofr current batch
            batch_key_list = inst_key_list[batch_n * batch_size:(batch_n + 1) * batch_size]
            # get and accumulate loss for batch instances:
            for instance in batch_key_list:
                hidden_states = torch.FloatTensor(data[instance]['last_hidden_states'][()]).to(pt_device)
                # one-hot labels as FloatTensor for CELoss:
                labels = torch.FloatTensor(data[instance]['tkn_lbls'][()]).to(pt_device)
                # get prediction:
                pred = model(hidden_states)
                # calculate loss:
                loss = loss_fct(pred.view(-1, num_labels), labels)
                loss_list.append(loss.item())
                # scale loss by batch size:
                loss = loss / batch_size
                # update gradients:
                loss.backward()

            # optimize for each batch:
            optimizer.step()

        print(f"Average loss for epoch {cur_epoch}: {mean(loss_list)}")
        # save checkpoints:
        if epochs_per_ckpt:
            if cur_epoch % epochs_per_ckpt == 0:
                if save_path:
                    print(f"Saving epoch {cur_epoch} checkpoint.")
                    torch.save({'epoch': cur_epoch, 'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(), 'lr': lr, 'loss': mean(loss_list),
                                'batch_size': batch_size, 'save_name': save_name, 'drop_p': drop_p},
                               f"{save_path}/{save_name}_epoch{cur_epoch}.ckpt")
        # iterate epoch:
        cur_epoch += 1

    if save_final_epoch_ckpt:
        if save_path:
            print(f"Saving epoch {cur_epoch} checkpoint.")
            torch.save({'epoch': cur_epoch - 1, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'lr': lr, 'loss': mean(loss_list),
                        'batch_size': batch_size, 'save_name': save_name, 'drop_p': drop_p},
                       f"{save_path}/{save_name}_epoch{cur_epoch}.ckpt")

    # save trained model:
    if save_path:
        torch.save(model.state_dict(), f"{save_path}/{save_name}.pt")


def train_simple_linear_head(train_data_path: str, save_path: str = '', save_name: str = 'simpleLinearLNERHead',
                             epochs_per_ckpt: int = 0, save_final_epoch_ckpt: bool = False, batch_size: int = 1,
                             lr: float = 0.001, epochs: int = 3):
    """
    Train a SimpleLinearLNERHead on the full training data.

    :param train_data_path: Path of the training data.
    :param save_path: Path of directory to save trained model file at. If omitted, model will not be saved! Pass . to
    save in current directory.
    :param save_name: Name for the saved model file(s).
    :param epochs_per_ckpt: Save a checkpoint every epochs_per_ckpt epochs. If 0, no checkpoints will be saved.
    :param save_final_epoch_ckpt: Save checkpoint after last epoch of training run.
    :param batch_size: Size of training batches for batch GD. Gradients are applied for each batch. 1 = SGD
    :param lr: Learning rate.
    :param epochs: Number of epochs.
    :return: Trained model.
    """
    training_model = SimpleLinearLNERHead(drop_p=0.0).to(pt_device)
    training_data = h5py.File(train_data_path, 'r')
    # train model:
    simple_linear_head_train_loop(training_model, training_data,
                                  save_path=save_path, save_name=save_name,
                                  epochs_per_ckpt=epochs_per_ckpt, save_final_epoch_ckpt=save_final_epoch_ckpt,
                                  batch_size=batch_size, lr=lr, epochs=epochs, drop_p=0.0)
    return training_model


def train_simple_linear_head_from_ckpt(train_data_path: str, ckpt_path: str, save_path: str = '',
                                       epochs_per_ckpt: int = 0, save_final_epoch_ckpt: bool = False,
                                       epochs: int = 3):
    """
    Continue training a SimpleLinearLNERHead on the full training data from a saved checkpoint. Training
    hyperparameters are loaded from checkpoint file.

    :param train_data_path: Path of the training data.
    :param ckpt_path: Path of the checkpoint to continue from.
    :param save_path: Path of directory to save trained model file at. If omitted, model will not be saved! Pass . to
    save in current directory.
    :param epochs_per_ckpt: Save a checkpoint every epochs_per_ckpt epochs. If 0, no checkpoints will be saved.
    :param save_final_epoch_ckpt: Save checkpoint after last epoch of training run.
    :param epochs: Number of epochs.
    :return: Trained model.
    """
    training_model = SimpleLinearLNERHead(drop_p=0.0).to(pt_device)
    # load checkpoint data:
    checkpoint = torch.load(ckpt_path)
    # apply checkpoint data to training model instance:
    training_model.load_state_dict(checkpoint['model_state_dict'])

    # basic Adam optimizer:
    checkpoint_optimizer = torch.optim.Adam(training_model.parameters(), lr=checkpoint['lr'])
    # apply checkpoint data to optimizer instance:
    checkpoint_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # initialize model in train mode:
    training_model.train()

    training_data = h5py.File(train_data_path, 'r')
    # train model:
    simple_linear_head_train_loop(training_model, training_data,
                                  ckpt_optimizer=checkpoint_optimizer, initial_epoch=checkpoint['epoch']+1,
                                  save_path=save_path, save_name=checkpoint['save_name'],
                                  epochs_per_ckpt=epochs_per_ckpt, save_final_epoch_ckpt=save_final_epoch_ckpt,
                                  batch_size=checkpoint['batch_size'], epochs=epochs)
    return training_model


def train_simple_dropout_linear_head(train_data_path: str, save_path: str = '', save_name: str = 'simpleLinearLNERHead',
                                     epochs_per_ckpt: int = 0, save_final_epoch_ckpt: bool = False, batch_size: int = 1,
                                     lr: float = 0.001, epochs: int = 3, drop_p: float = 0.5):
    """
    Train a SimpleLinearLNERHead with dropout on the full training data.

    :param train_data_path: Path of the training data.
    :param save_path: Path of directory to save trained model file at. If omitted, model will not be saved! Pass . to
    save in current directory.
    :param save_name: Name for the saved model file(s).
    :param epochs_per_ckpt: Save a checkpoint every epochs_per_ckpt epochs. If 0, no checkpoints will be saved.
    :param save_final_epoch_ckpt: Save checkpoint after last epoch of training run.
    :param batch_size: Size of training batches for batch GD. Gradients are applied for each batch. 1 = SGD
    :param lr: Learning rate.
    :param epochs: Number of epochs.
    :param drop_p: Dropout probability.
    :return: Trained model.
    """
    training_model = SimpleLinearLNERHead(drop_p=drop_p).to(pt_device)
    training_data = h5py.File(train_data_path, 'r')
    # train model:
    simple_linear_head_train_loop(training_model, training_data,
                                  save_path=save_path, save_name=save_name,
                                  epochs_per_ckpt=epochs_per_ckpt, save_final_epoch_ckpt=save_final_epoch_ckpt,
                                  batch_size=batch_size, lr=lr, epochs=epochs, drop_p=drop_p)
    return training_model


def train_simple_dropout_linear_head_from_ckpt(train_data_path: str, ckpt_path: str, save_path: str = '',
                                               epochs_per_ckpt: int = 0, save_final_epoch_ckpt: bool = False,
                                               epochs: int = 3):
    """
    Continue training a SimpleLinearLNERHead on the full training data from a saved checkpoint. Training
    hyperparameters are loaded from checkpoint file.

    :param train_data_path: Path of the training data.
    :param ckpt_path: Path of the checkpoint to continue from.
    :param save_path: Path of directory to save trained model file at. If omitted, model will not be saved! Pass . to
    save in current directory.
    :param epochs_per_ckpt: Save a checkpoint every epochs_per_ckpt epochs. If 0, no checkpoints will be saved.
    :param save_final_epoch_ckpt: Save checkpoint after last epoch of training run.
    :param epochs: Number of epochs.
    :return: Trained model.
    """
    # load checkpoint data:
    checkpoint = torch.load(ckpt_path)

    training_model = SimpleLinearLNERHead(drop_p=checkpoint['drop_p']).to(pt_device)

    # apply checkpoint data to training model instance:
    training_model.load_state_dict(checkpoint['model_state_dict'])

    # basic Adam optimizer:
    checkpoint_optimizer = torch.optim.Adam(training_model.parameters(), lr=checkpoint['lr'])
    # apply checkpoint data to optimizer instance:
    checkpoint_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # initialize model in train mode:
    training_model.train()

    training_data = h5py.File(train_data_path, 'r')
    # train model:
    simple_linear_head_train_loop(training_model, training_data,
                                  ckpt_optimizer=checkpoint_optimizer, initial_epoch=checkpoint['epoch']+1,
                                  save_path=save_path, save_name=checkpoint['save_name'],
                                  epochs_per_ckpt=epochs_per_ckpt, save_final_epoch_ckpt=save_final_epoch_ckpt,
                                  batch_size=checkpoint['batch_size'], epochs=epochs, drop_p=checkpoint['drop_p'])
    return training_model


if __name__ == "__main__":
    # training data path:
    train_dataset_path = "./data/roberta_inference_full_train.hdf5"

    train_simple_dropout_linear_head(train_dataset_path, save_name="linearDropoutHead_3batch_demo", save_path=".",
                                     batch_size=3, epochs=10, save_final_epoch_ckpt=True)

    # train_simple_dropout_linear_head_from_ckpt(train_dataset_path, save_path=".",
    #                                           ckpt_path="linearDropoutHead_3batch_epoch10.ckpt",
    #                                           epochs=30, epochs_per_ckpt=10, save_final_epoch_ckpt=True)

    # train_simple_dropout_linear_head_from_ckpt(train_dataset_path, save_path=".",
    #                                           ckpt_path="linearDropoutHead_3batch_epoch40.ckpt",
    #                                           epochs=30, epochs_per_ckpt=10, save_final_epoch_ckpt=True)

    # train_simple_linear_head(train_dataset_path, save_name="linearHead_3batch", save_path=".",
    #                         batch_size=3, epochs=10, save_final_epoch_ckpt=True)

    # train_simple_linear_head_from_ckpt(train_dataset_path, save_path=".",
    #                                           ckpt_path="linearHead_3batch_epoch10.ckpt",
    #                                           epochs=30, epochs_per_ckpt=10, save_final_epoch_ckpt=True)

    # train_simple_linear_head_from_ckpt(train_dataset_path, save_path=".",
    #                                   ckpt_path="linearHead_3batch_epoch40.ckpt",
    #                                   epochs=30, epochs_per_ckpt=10, save_final_epoch_ckpt=True)

    # print("Final loss on lin dropout:", torch.load("linearDropoutHead_3batch_epoch70.ckpt")['loss'])
    # print("Final loss on lin no-drop:", torch.load("linearHead_3batch_epoch70.ckpt")['loss'])
