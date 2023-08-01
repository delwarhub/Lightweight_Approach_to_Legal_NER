"""Main model for token label inference"""

from typing import Optional, Tuple

import torch

from transformers import RobertaPreTrainedModel, RobertaModel, RobertaTokenizerFast, AutoConfig, logging

from model.linear_heads import SimpleLinearLNERHead

from model.convolution_heads import ConvolutionLNERHeadShallow, ConvolutionLNERHeadDeep

from data.data_util import convert_lblid


# load tokenizer:
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
# load RoBERTa-base HF config:
config = AutoConfig.from_pretrained("roberta-base")
# don't show HF transformers 'missing head' warning when loading RoBERTa:
logging.set_verbosity_error()

# get torch device:
pt_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model based on
# https://github.com/huggingface/transformers/blob/v4.27.0/src/transformers/models/roberta/modeling_roberta.py#L1360


class RobertaLegalNER(RobertaPreTrainedModel):
    """
    Model class for inference, using a specified head.
    Pass head class 'linear', 'conv_shallow' or 'conv_deep' and path to saved head model instance when initializing.
    """
    def __init__(self, config, head_class: str, head_path: str):
        super().__init__(config)
        # load RoBERTa model:
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        # create specified head class instance:
        if head_class == "linear":
            # use linear head model class; dropout probability 0.0 for inference:
            head_model = SimpleLinearLNERHead(drop_p=0.0)
        elif head_class == "conv_shallow":
            # use shallow convolution head model class:
            head_model = ConvolutionLNERHeadShallow()
        elif head_class == "conv_deep":
            # use deep convolution head model class:
            head_model = ConvolutionLNERHeadDeep()
        else:
            raise ValueError(f'Unrecognized head "{head_class}"; options: "linear", "conv_shallow" or "conv_deep".')
        # load head model data:
        head_model.load_state_dict(torch.load(head_path))
        head_model.eval()

        self.classifier = head_model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None
    ) -> Tuple[torch.Tensor]:

        outputs = self.roberta(input_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output)

        return logits


def infer_text_labels(text: str, head_class: str, head_path: str, verbose: bool = False) -> list:
    """
    Infer token NER labels for any text.

    :param text: Input text.
    :param head_class: Model head class to be used; 'linear', 'conv_shallow' or 'conv_deep'.
    :param head_path: Path of saved model head instance.
    :param verbose: print input text, token list, and inferred token labels
    :return: Token label list.
    """
    if verbose:
        print(f"Inferring token labels using {head_class} head loaded from {head_path}.")
        print("Input text:", text)
    with torch.no_grad():
        # initialize model:
        inference_model = RobertaLegalNER(config, head_class=head_class, head_path=head_path).to(pt_device)
        # tokenize input text:
        text_tkns = tokenizer(text)
        # create list of readable tokens:
        token_list = []
        for input_id in text_tkns['input_ids']:
            token_list.append(tokenizer.decode(input_id))
        # show readable tokens:
        if verbose:
            print(token_list)
        # convert tokens and attention mask to fit model:
        tensor_input_ids = torch.unsqueeze(torch.LongTensor(text_tkns['input_ids']), 0).to(pt_device)
        tensor_attention_mask = torch.unsqueeze(torch.LongTensor(text_tkns['attention_mask']), 0).to(pt_device)
        # infer token label logits:
        inference = inference_model(tensor_input_ids, attention_mask=tensor_attention_mask)
        # create list of inferred token labels:
        token_label_list = []
        for token_i in range(inference[0].size()[0]):
            token_label_id = inference[0][token_i].argmax()
            token_label_id = token_label_id.item()
            token_label_list.append(convert_lblid(token_label_id))
        # show inferred token labels:
        if verbose:
            print(token_label_list)
    return token_label_list


if __name__ == "__main__":

    test_text = "The amendments made in Water Act, 1988 have not been adopted in Gujarat."

    # infer_text_labels(test_text, "linear", "linearDropoutHead_3batch.pt", verbose=True)
    infer_text_labels(test_text, "linear", "linearHead_3batch.pt", verbose=True)
    # infer_text_labels(test_text, "conv_shallow", "shallowConvHead_3batch.pt", verbose=True)
    # infer_text_labels(test_text, "conv_deep", "deepConvHead_3batch.pt", verbose=True)


