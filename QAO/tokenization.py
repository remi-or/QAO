from typing import List, Dict, Tuple
import pandas as pd
from transformers import AutoTokenizer



DataFrame = pd.DataFrame
Tokens = List[int]
QaKey = Tuple[int, int]



def tokenize_qa_couples(
    formatted_dataframe : DataFrame,
    tokenizer_name : str = 'camembert-base',
    use_questions : bool = True,
    use_answers : bool = True,
) -> Dict[QaKey, Tokens]:
    """
    Tokenizes the inputs contained in a (formatted_dataframe) like the one returned by [extraction.extract_qa_couples].
    Uses an AutoTokenizer loaded from a pre trained one with the given (tokenizer_name).
    If the (use_questions) flag is set to True, the question is tokenized.
    If the (use_answers) flag is set to True, the answer is tokenized, after the question if it is also tokenize.
    Returns a dictionnary containing the tokenized qa couples with the couple's key as key.
    """
    # Load the keys
    keys = list(formatted_dataframe.loc[:, 'key'])
    # Load the inputs according to the arguments
    inputs = [
        f"{row['question'].strip() + ' ' if use_questions else ''}{row['answer'].strip() if use_answers else ''}"
        for _, row in formatted_dataframe.iterrows()
    ]
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # Actual tokenization
    tokenization_kwargs = {'padding' : False, 'truncation' : False, 'return_attention_mask' : False}
    tokenized = tokenizer(inputs, **tokenization_kwargs)['input_ids']
    # Return the key : tokens where tokens doesn't have a start or end token_id
    return {key : tokens[1 : -1] for key, tokens in zip(keys, tokenized)}


def tokenize_objectives(
    objectives_dictionnary : Dict[str, str],
    tokenizer_name : str = 'camembert-base',
) -> Dict[str, Tokens]:
    """
    Tokenizes the objectives contained in an (objectives_dictionnary) like the one returned by [extraction.extract_objectives]
    Uses an AutoTokenizer loaded from a pre trained one with the given (tokenizer_name).
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # Actual tokenization
    tokenization_kwargs = {'padding' : False, 'truncation' : False, 'return_attention_mask' : False}
    tokenized = tokenizer(list(objectives_dictionnary.values()), **tokenization_kwargs)['input_ids']
    # Return the key : tokens where tokens doesn't have a start or end token_id
    return {key : tokens[1 : -1] for key, tokens in zip(objectives_dictionnary.keys(), tokenized)}