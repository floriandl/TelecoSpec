import dspy

class TeleQnABaseSignature(dspy.Signature):
    """
    Answer the given question by choosing the correct option from the provided choices.
    Return the exact answer without any additional text or explanation. (e.g. "choice N: <option text>") 
    """
    question: str = dspy.InputField(description="The question to be answered.")
    choices: dict = dspy.InputField(description="A dictionary containing the options for the question.")
    answer: str = dspy.OutputField(description="Your answer to the question.")

class TeleQnARAGSignature(dspy.Signature):
    """
    Answer the given question by choosing the correct option from the provided choices.
    Return the exact answer without any additional text or explanation. (e.g. "choice N: <option text>") 
    """
    context: str = dspy.InputField(description="The context to be used for answering the question.")
    question: str = dspy.InputField(description="The question to be answered.")
    choices: dict = dspy.InputField(description="A dictionary containing the options for the question.")
    answer: str = dspy.OutputField(description="Your answer to the question.")