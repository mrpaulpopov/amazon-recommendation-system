from pydantic import BaseModel, Field

class TopkModel(BaseModel):
    user_str: str = "A3PHJ4NMHMBBUB"
    top_k: int = Field(10, ge=1)