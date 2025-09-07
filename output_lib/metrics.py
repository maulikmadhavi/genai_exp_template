from typing import List, Union


def matching_scores(predictions: List[Union[str, int]], references: List[Union[str, int]]) -> float:
    """Calculate the matching score between predictions and references."""
    correct = sum(p == r for p, r in zip(predictions, references))
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")
    return float(correct) / len(references) if references else 0.0


def matching_dict(predictions: List[Union[str, int]], references: List[Union[str, int]]) -> dict:
    """Calculate the matching score for each unique label."""
    score_dict = {}
    unique_labels = set(references)
    # Must have  the same keys in predictions and references
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")
    sum_scores = 0.0
    for label in unique_labels:
        label_preds = [p for p, r in zip(predictions, references) if r == label]
        label_refs = [r for r in references if r == label]
        score_dict[label] = matching_scores(label_preds, label_refs)
        sum_scores += score_dict[label]
    score_dict["overall"] = sum_scores / len(unique_labels) if unique_labels else 0.0

    return score_dict
