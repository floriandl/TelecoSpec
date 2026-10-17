import dspy

from src.models.dspy.signatures import TeleQnABaseSignature


class TeleQnABaseModel(dspy.Module):
    """
    A model that answers questions based on the provided choices.
    """
    def __init__(self, signature:dspy.Signature = TeleQnABaseSignature, llm: dspy.LM = None):
        self.candidate = dspy.Predict(signature=signature)
        self.llm = llm

    def forward(self, question: str, choices: dict) -> str:
        with dspy.context(lm=self.llm):
            result = self.candidate(question=question, choices=choices)
        return result

# class TeleQnAAgentModel(dspy.Module):
#     """
#     A model that answers questions based on the provided choices.
#     """
#     def __init__(self, signature:dspy.Signature = TeleQnARAGSignature, llm: dspy.LM = None, tools: list[dspy.Tool] = None):
#         self.candidate = dspy.ReAct(signature=signature, tools=tools)
#         self.llm = llm

#     def forward(self,  question: str, choices: dict) -> str:
#         context = _bm25_search(question)
#         with dspy.context(lm=self.llm):
#             result = self.candidate(context=context, question=question, choices=choices)
#         return result