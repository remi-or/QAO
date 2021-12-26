from typing import List, Dict, Tuple
import pandas as pd
import numpy as np



Tokens = List[int]
QaKey = Tuple[int, int]
ScoredId = Tuple[str, float]
DataFrame = pd.DataFrame



def full_format_data(
    keys_and_two_best : Dict[QaKey, Tuple[ScoredId, ScoredId]],
    qa_couples : DataFrame,
    objectives : Dict[str, str],
) -> DataFrame:
    # Copy the original Qa_couples dataframe
    full_dataframe = qa_couples.copy()
    # Main loop
    for i, row in full_dataframe.iterrows():
        # Get the qa key of the row
        qa_key = row['key']
        # Add the qid and aid columns
        full_dataframe.loc[i, 'question id'] = qa_key[0]
        full_dataframe.loc[i, 'answer id'] = qa_key[1]
        # Retrieve the best and aux
        best, aux = keys_and_two_best[qa_key]
        # Add the best-related columns
        full_dataframe.loc[i, ['best PO', 'best PO id', 'best PO score']] = [objectives[best[0]], best[0], best[1]]
        # Add the aux-related columns
        full_dataframe.loc[i, ['aux PO', 'aux PO id', 'aux PO score']] = [objectives[aux[0]], aux[0], aux[1]]
    # Drop the key column and return
    return full_dataframe.drop(columns=['key'])