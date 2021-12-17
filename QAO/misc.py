from typing import Union, List, Optional
import torch
from transformers import AutoTokenizer



Tensor = torch.Tensor
Number = Union[int, float]
Tokens = List[int]



def pad_and_tensorize(
    list_of_lists : List[List[Number]],
    padding : Number = 0,
    max_length : Optional[int] = None, 
    return_attention_mask : bool = True,
    to_cuda : bool = False,
    pad_to_max_length : bool = False,
    ) -> List[List[Number]]:
    """
    Pads a (list_of_list) with a given (padding) and returns it as a tensor.
    If a (max_length) argument is passed, the tensor dim=1 may be truncated.
    If the (return_attention_mask) is passed, also returns the attention masks. 
    """
    # Compute inner list max length
    inner_list_max_length = max(len(inner_list) for inner_list in list_of_lists)
    # If a max length is passed, ensures it's respected
    if max_length is not None:
        if pad_to_max_length:
            max_length = max_length
        else:
            max_length = min(max_length, inner_list_max_length)
    else:
        max_length = inner_list_max_length
    # The tensor that's going to be returned
    tensor = torch.full(
        size=(len(list_of_lists), max_length),
        fill_value=padding,
    )
    # Eventual attention mask
    if return_attention_mask:
        attention_mask = torch.zeros(
            size=(len(list_of_lists), max_length),
            dtype=torch.int64,
        )
    # Inner lists loop
    for i, inner_list in enumerate(list_of_lists):
        # Compute inner list length
        inner_list_length = min(len(inner_list), max_length)
        # Place the inner list in the returned tensor
        tensor[i, :inner_list_length] = torch.tensor(inner_list[:inner_list_length])
        # Eventually place the inner list attention mask
        if return_attention_mask:
            attention_mask[i, :inner_list_length] = torch.ones(size=(inner_list_length,))
    # Eventual to cuda
    if to_cuda:
        tensor = tensor.to('cuda')
        attention_mask = attention_mask.to('cuda') if return_attention_mask else None
    # Returns
    if return_attention_mask:
        return tensor, attention_mask
    else:
        return tensor


def retrieve_special_tokens(
    tokenizer_name : str = 'camembert-base',
    ) -> List[int]:
    """
    Retrieves the tokens used in tokenization of a couple of sentences.
    Uses an AutoTokenizer loaded from a pre trained one with the given (tokenizer_name).
    Returns a list of four tokens, in order : bos, eos-sep, bos-sep, eos.
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # Return the relevant tokens
    return tokenizer(' ', ' ', return_attention_mask=False)['input_ids']