from langchain_community.llms import HuggingFacePipeline  # Mise à jour de l'import
from callback import MyCallbackHandler
from langchain.callbacks.base import BaseCallbackManager
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class QaLlm():
    def __init__(self) -> None:
        # Chargement du modèle LLaMA 2
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")

        # Création du pipeline pour l'inférence
        llama_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=500)

        # Intégration dans LangChain
        manager = BaseCallbackManager([MyCallbackHandler()])
        self.llm = HuggingFacePipeline(pipeline=llama_pipeline, callback_manager=manager)

    def get_llm(self):
        return self.llm

# # Test rapide
# if __name__ == "__main__":
#     qa_llm = QaLlm()
#     print(qa_llm.get_llm()("Quels sont les symptômes de l'insuffisance rénale ?"))