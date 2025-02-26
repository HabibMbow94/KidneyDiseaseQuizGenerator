from qcm_chain import QCMGenerateChain
from qa_llm import QaLlm
from langchain.output_parsers.regex import RegexParser
from typing import List
import asyncio

# Définition des parsers pour extraire les questions et réponses
parsers = {
    "question": RegexParser(
        regex=r"question:\s*(.*?)\s+(?:\n)+",
        output_keys=["question"]
    ),
    "A": RegexParser(
        regex=r"(?:\n)+\s*CHOICE_A:(.*?)\n+",
        output_keys=["A"]
    ),
    "B": RegexParser(
        regex=r"(?:\n)+\s*CHOICE_B:(.*?)\n+",
        output_keys=["B"]
    ),
    "C": RegexParser(
        regex=r"(?:\n)+\s*CHOICE_C:(.*?)\n+",
        output_keys=["C"]
    ),
    "D": RegexParser(
        regex=r"(?:\n)+\s*CHOICE_D:(.*?)\n+",
        output_keys=["D"]
    ),
    "reponse": RegexParser(
        regex=r"(?:\n)+reponse:\s?(.*)",
        output_keys=["reponse"]
    )
}


qa_llm = QaLlm()
qa_chain = QCMGenerateChain.from_llm(qa_llm.get_llm())

async def llm_call(qa_chain: QCMGenerateChain, texts: List[str]):
    """
    Appelle le LLM pour générer les questions à partir des textes.
    """
    print(f"🛠️ LLM en cours d'exécution...")

    # Utilisation de `predict_batch()` pour traiter les textes en parallèle
    batch_examples = await asyncio.gather(*[qa_chain.apredict(text=text) for text in texts])

    print(f"✅ LLM terminé.")

    return batch_examples


async def generate_quizz(contents: List[str]):
    """
    Génère un quiz à partir des contenus fournis.
    """
    docs = [{"text": content} for content in contents]

    return await llm_call(qa_chain, docs)  
    