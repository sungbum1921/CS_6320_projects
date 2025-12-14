from pydantic import BaseModel

class QueryInput(BaseModel):
    query: str
    model_id: str = "t5_small"


class QueryResponse(BaseModel):
    message: str
    timestamp: str
    type: str
