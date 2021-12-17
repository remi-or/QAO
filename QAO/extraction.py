import pickle
from typing import Dict, Tuple
import pandas as pd



DataFrame = pd.DataFrame
QaKey = Tuple[int, int]



def load_dataframe(
    filepath : str,
    encoding : str = 'utf-8',
) -> DataFrame:
    if filepath.endswith('.csv'):
        with open(filepath, 'r', encoding=encoding) as file:
            return pd.read_csv(file)
    else:
        raise(ValueError(f"Unsupported extension: {filepath}"))



def extract_qa_couples(
    filepath : str,
    question_column : str = 'question',
    question_id_column : str = 'question_id',
    answer_column : str = 'answer',
    answer_id_column : str = 'answer_id',
) -> DataFrame:
    # We're going to rename the relevant column
    rename_dict = {
        question_column : 'question',
        answer_column : 'answer',
        'key' : 'key',
    }
    # Load the dataframe
    dataframe = load_dataframe(filepath)
    # Add the id column
    dataframe.loc[:, 'key'] = [key for key in zip(dataframe.loc[:, question_id_column], dataframe.loc[:, answer_id_column])]
    # Rename the columns
    dataframe = dataframe.rename(columns=rename_dict)
    # Drop the unrelevant colomns
    dataframe = dataframe.drop(columns=[column for column in dataframe.columns if (column not in rename_dict.values())])
    # Drop the NaNs
    dataframe = dataframe.dropna()
    # Return
    return dataframe


def extract_objectives(
    filepath : str,
    id_column : str = 'ID long OP',
    description_column : str = 'Objectif',
) -> Dict[str, str]:
    """
    Extracts the objectives saved in the file pointed to by (filepath).
    The objective's ids are in the (id_column) and the text describing the said objectives is in the (description_column).
    Returns the results as a dict where id : description.
    """
    return {row[id_column] : row[description_column] for _, row in load_dataframe(filepath).iterrows()}


def get_number_of_logged_keys(
    log_filepath : str,
) -> int:
    number_of_logged_keys = 0
    running = True
    with open(log_filepath, 'rb') as file:
        while True:
            try:
                x = pickle.load(file)
                number_of_logged_keys += len(x)
            except EOFError:
                break
    return number_of_logged_keys