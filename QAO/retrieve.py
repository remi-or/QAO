from typing import List, Dict, Tuple
import json

from QAO.iterator import key_qao_iterator



Tokens = List[int]
QaKey = Tuple[int, int]
ScoredId = Tuple[str, float]



def associate_keys_and_two_best(
    tokenized_qa_couples : Dict[QaKey, Tokens],
    tokenized_objectives : Dict[str, Tokens],
    log_filepath : str,
) -> Dict[QaKey, Tuple[ScoredId, ScoredId]]:
    # Retrieve the scores
    with open(log_filepath) as file:
        scores = json.load(file)
    # First loop
    qa_to_scored_ids = {}
    for (qa_key, obj_id), score in zip(key_qao_iterator(tokenized_qa_couples, tokenized_objectives), scores):
        if qa_key not in qa_to_scored_ids:
            qa_to_scored_ids[qa_key] = [(obj_id, score)]
        else:
            qa_to_scored_ids[qa_key].append((obj_id, score))
    # Second loop
    qa_to_best_scored_ids, sorting_function = {}, (lambda x : x[1])
    for qa_key, scored_ids in qa_to_scored_ids.items():
        scored_ids.sort(key=sorting_function, reverse=True)
        qa_to_best_scored_ids[qa_key] = tuple(scored_ids[:2])
    return qa_to_best_scored_ids