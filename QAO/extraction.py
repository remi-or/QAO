import json
from typing import Dict, Tuple
import pandas as pd



DataFrame = pd.DataFrame
QaKey = Tuple[int, int]



def load_dataframe(
    filepath : str,
    encoding : str = 'utf-8',
) -> DataFrame:
    """
    Loads a dataframe saved in a .csv file pointed to by (filepath).
    Loads it with the specified (encoding), utf-8 by default.
    """
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
    source_column : str = 'source',
) -> DataFrame:
    """
    Extracts the question-answers couples from a .csv file pointed to by (filepath).
    There is expected to be at least 4 columns, with the names:
        (question_column) for the questions' text
        (question_id_column) for the questions' id
        (answer_column) for the answers' text
        (answer_id_column) for the answers' id
        (source_column) for the question's source
    Returns a DataFrame with standardized 4 columns, question, answer, key, where key = (question_id, answer_id), and source.
    """
    # We're going to rename the relevant column
    rename_dict = {
        question_column : 'question',
        answer_column : 'answer',
        'key' : 'key',
        source_column : 'source',
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
    """
    In a log file pointed to by (log_filepath) created by QAO.score's [score_qa_objective_couples], retrieves the last key logged in.
    This enables the scoring to fail at some point and restart from the same point.
    """
    with open(log_filepath, 'r') as file:
        scores_accumulator = json.load(file)
    return len(scores_accumulator)


def extract_source_to_document(
    filepath : str,
    source_column : str = 'source',
    document_column : str = 'document',
) -> Dict[str, str]:
    """
    Loads a source to document dictionnary saved in a .csv file pointed to by (filepath).
    The sources are in the column named (source_column) and same goes for document in (document_column).
    """
    return {row[source_column] : row[document_column] for _, row in load_dataframe(filepath).iterrows()}


def extract_document_to_days(
    filepath : str,
    document_column : str = 'document',
    day_column : str = 'day',
) -> Dict[str, str]:
    """
    Converts a documents dataframe located at (filepath) to a dictionnary.
    The documents are stored in (document_column) and days stored in (day_column).
    """
    return {row[document_column] : row[day_column] for _, row in load_dataframe(filepath).iterrows()}