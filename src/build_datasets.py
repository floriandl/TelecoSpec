import dspy

from src.data_processing import load_json

def build_examples(set_path: str, limit: int = None) -> list[dspy.Example]:
    """
    Build examples from the TeleQnA dataset for the specified release number.
    """
    data = load_json(set_path)
    examples = []
    for qcm in data:
        question = qcm.get("question", "")
        choices = {
            "option 1": qcm.get("option 1", ""),
            "option 2": qcm.get("option 2", ""),
            "option 3": qcm.get("option 3", ""),
            "option 4": qcm.get("option 4", ""),
            "option 5": qcm.get("option 5", ""),
        }
        answer = qcm.get("answer", "")
        explanation = qcm.get("explanation", "")
        example = dspy.Example(
            question=question,
            choices=choices,
            answer=answer,
            explanation=explanation,
        ).with_inputs("question", "choices")
        examples.append(example)
    if limit:
        return examples[:limit]
    return examples

if __name__ == "__main__":
    set_path = "data/processed/TeleQnA_Rel17_dev.json"
    examples = build_examples(set_path=set_path, limit=5)
    for example in examples:
        print(example.question)
