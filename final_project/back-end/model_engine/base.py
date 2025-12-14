from abc import ABC, abstractmethod

class BaseSQLModel(ABC):
    @abstractmethod
    def load(self, model_path: str):
        """모델과 토크나이저를 로드합니다."""
        pass

    @abstractmethod
    def generate_sql(self, input_text: str) -> str:
        """자연어를 받아 SQL을 반환합니다."""
        pass
