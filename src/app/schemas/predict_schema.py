from pydantic import BaseModel

class PredictModel(BaseModel):
    user_str: str = "A1GYLH2VMW2MQI"
    item_strs: list[str] = ["B00NIYJL64", "B00BKVQETY", "B016I3T3CI"]