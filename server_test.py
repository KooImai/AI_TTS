from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Any, Dict
import uvicorn
import os

AIP_HEALTH_ROUTE = os.environ.get('AIP_HEALTH_ROUTE', '/health')
AIP_PREDICT_ROUTE = os.environ.get('AIP_PREDICT_ROUTE', '/predict')

app = FastAPI()

class PredictionRequest(BaseModel):
    instances: List[Any]
    parameters: Dict[str, Any]

class PredictionResponse(BaseModel):
    predictions: List[Any]

@app.get(AIP_HEALTH_ROUTE, status_code=200)
async def health():
    return {'health': 'ok'}

@app.post(AIP_PREDICT_ROUTE, response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # ここで予測ロジックを実装します。以下はダミーの予測結果です。
    predictions = [{"result": instance} for instance in request.instances]
    
    return PredictionResponse(predictions=predictions)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)