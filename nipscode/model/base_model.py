from abc import ABC, abstractmethod
from typing import Optional, Any

class BaseModel(ABC):
    @abstractmethod
    def infer(self, user_prompt: str, **kwargs: Any) -> Optional[str]:
        pass
