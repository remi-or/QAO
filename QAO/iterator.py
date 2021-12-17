from typing import List, Dict, Optional, Tuple
from itertools import product

from QAO.misc import retrieve_special_tokens



Tokens = List[int]
QaKey = Tuple[int, int]
QaoKey = Tuple[QaKey, str]



def key_qao_iterator(
    tokenized_qa_couples : Dict[QaKey, Tokens],
    tokenized_objectives : Dict[str, Tokens],
    keys_to_skip : int = 0,
) -> QaoKey:
    iterator = product(tokenized_qa_couples.keys(), tokenized_objectives.keys())
    for i in range(keys_to_skip):
        _= next(iterator)
    return iterator


def tokenized_qao_iterator(
    tokenized_qa_couples : Dict[QaKey, Tokens],
    tokenized_objectives : Dict[str, Tokens],
    batch_size : int = 16,
    tokenizer_name : str = 'camembert-base',
    objective_first : bool = True,
    keys_to_skip : int = 0,
    ) -> List[Tokens]:
    """
    Iterates over every couple of qa and objectives described by (tokenized_qa_couples) and (tokenized_objectives) and returns the formatted input.
    Yields batches of (batch_size)
    Uses the special tokens that a tokenizer with (tokenizer_name) would to form the coupled inputs.
    """
    # Get the special tokens
    special_tokens = retrieve_special_tokens(tokenizer_name)
    # Create the input formatting function depending on the order
    if objective_first:
        format_input = lambda qa, objective : [special_tokens[0]] + objective + special_tokens[1:3] + qa + [special_tokens[3]]
    else:
        format_input = lambda qa, objective : [special_tokens[0]] + qa + special_tokens[1:3] + objective + [special_tokens[3]]
    # Accumulators for the main loop
    current_batch_inputs, current_batch_size = [], 0 
    # Main loop
    for qa_key, objective_id in key_qao_iterator(tokenized_qa_couples, tokenized_objectives, keys_to_skip):
        # Add the values to the accumulators
        current_batch_inputs.append(format_input(tokenized_qa_couples[qa_key], tokenized_objectives[objective_id]))
        current_batch_size += 1
        # Yield if the batch is big enough
        if current_batch_size == batch_size:
            yield  current_batch_inputs
            current_batch_inputs, current_batch_size = [], 0 
    # Yields the leftovers
    if current_batch_size > 0:
        yield current_batch_inputs


def get_number_of_batches(
    tokenized_qa_couples : Dict[QaKey, Tokens],
    tokenized_objectives : Dict[str, Tokens],
    batch_size : int = 16,
    keys_to_skip : int = 0,
) -> int:
    total_size = 0
    for _ in key_qao_iterator(tokenized_qa_couples, tokenized_objectives, keys_to_skip):
        total_size += 1
    return (total_size // batch_size) + int(total_size % batch_size != 0)