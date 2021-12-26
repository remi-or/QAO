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
    """
    Given a (log_filepath) pointing to the log of a finished run of QAO.score's [score_qa_objective_couples],
    a dictionnary of (tokenized_qa_couples) like the one returned by QAO.tokenization's [tokenize_qa_couples],
    a dictionnary of (tokenized_objectives) like the one returned by QAO.tokenazition's [tokenized_objectives],
    returns a dictionnary containing for each qa_couple the best and second best objective associated with it.
    The returned dictionnary represents each qa_couple by its qa_key (question_id, answer_id),
    and each objective by its objective_id and the associated score.
    """
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