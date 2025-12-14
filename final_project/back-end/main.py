from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import schemas, query_handler
import time


app = FastAPI()

# @app.on_event("startup")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your React app's origin
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def read_root():
    return {"API working!!"}

@app.post("/query/", response_model=schemas.QueryResponse)
def get_qeuery_response(input: schemas.QueryInput):
    try:
        print(input)
        response = query_handler.generate_sql(input.query, input.model_id)
        print(response)
        timestamp = str(int(time.time() * 1000))
        return schemas.QueryResponse(
            message=response,  # Assign SQL to `message` field
            timestamp=timestamp,
            type="response"
        )


    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

