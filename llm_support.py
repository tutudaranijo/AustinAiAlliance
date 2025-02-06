

from abc import ABC, abstractmethod

class BaseLLM(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class OpenAILLM(BaseLLM):
    """Implements an OpenAI-based LLM using LangChain’s ChatOpenAI."""
    def __init__(self, model_name: str):
        super().__init__(model_name)
        from langchain.chat_models import ChatOpenAI  # Ensure correct import
        self.llm = ChatOpenAI(model=model_name)

    def generate(self, prompt: str) -> str:
        # Use the .predict() method instead of .call()
        response_text = self.llm.predict(prompt)
        return response_text

# Implement other LLM classes similarly...

class LLMFactory:
    """Returns an instance of the correct LLM based on a provider name."""
    @staticmethod
    def create_llm(provider: str, model_name: str) -> BaseLLM:
        provider = provider.lower()
        if provider == "openai":
            return OpenAILLM(model_name)
        elif provider == "llama":
            return None #LlamaLLM(model_name)
        elif provider == "watsonx":
            return None # WatsonxLLM(model_name)
        elif provider == "deepsense":
            return None #DeepSenseLLM(model_name)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
