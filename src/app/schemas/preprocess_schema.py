from pydantic import BaseModel

class PreprocessData(BaseModel):
    samplesize: int = 100000