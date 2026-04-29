from pydantic import BaseModel

class TopkModel(BaseModel):
    user_str: str = "A3PHJ4NMHMBBUB"
    top_k: int = 10