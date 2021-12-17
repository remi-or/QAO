from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm
import json
from os.path import exists

from QAO.extraction import get_number_of_logged_keys
from QAO.iterator import get_number_of_batches, tokenized_qao_iterator
from QAO.misc import pad_and_tensorize



Tokens = List[int]
QaKey = Tuple[int, int]



def score_qa_objective_couples(
    model_name : str,
    tokenized_qa_couples : Dict[QaKey, Tokens],
    tokenized_objectives : Dict[str, Tokens],
    log_filepath : str,
    max_length : int = 512,
    batch_size : int = 16,
    tokenizer_name : str = 'camembert-base',
    objective_first : bool = True,
    save_freq : int = 100,
    ) -> None:
    """
    Scores every qa objectives couples described by (tokenized_qa_couples) and (tokenized_objectives).
    The scoring is done by a sentence classifier model like CamemBERT with the name (model_name).
    Saves the result in a .txt file pointed to by (log_filepath).
    Yields batches of (batch_size)
    Uses the special tokens that a tokenizer with (tokenizer_name) would to form the coupled inputs.
    If the (objective_first) flag is set to True, the objective goes into the question slot.
    """
    # Get the padding token
    padding = AutoTokenizer.from_pretrained(tokenizer_name).pad_token_id
    # Load the model and set it to cuda and eval
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.cuda()
    model.eval()
    # First key
    keys_to_skip = get_number_of_logged_keys(log_filepath) if exists(log_filepath) else 0
    if exists(log_filepath):
        with open(log_filepath, 'r') as file: 
            scores_accumulator = json.load(file)
    else:
        scores_accumulator = []
    # Main loop
    with torch.no_grad():
        iterator = tokenized_qao_iterator(tokenized_qa_couples, tokenized_objectives, batch_size, tokenizer_name, objective_first, keys_to_skip)
        total = get_number_of_batches(tokenized_qa_couples, tokenized_objectives, batch_size, keys_to_skip)
        tick = 0
        for inputs in tqdm(iterator, total=total):
            # Input formatting
            inputs = pad_and_tensorize(inputs, padding, max_length, to_cuda=True)
            # Prediction
            logits = model(*inputs).logits
            # Processing
            scores = logits.softmax(1)
            # Logging
            for score in scores[:, 1]:
                scores_accumulator.append(score.item())
            tick += 1
            if tick == save_freq:
                log_file, tick = open(log_filepath, 'w'), 0
                json.dump(scores_accumulator, log_file)
                log_file.close()