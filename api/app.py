# FastAPI应用
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from models.model_inference import Web3ModelInference
from config.inference_config import InferenceConfig
import uvicorn

# 创建FastAPI应用
app = FastAPI(
    title="Web3.0 AI Model API",
    description="API for interacting with Web3.0 domain-specific AI models",
    version="1.0.0"
)

# 加载配置
config = InferenceConfig()

# 初始化模型推理
model_inference = Web3ModelInference(config)


# 定义API请求模型
class PredictionRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = config.temperature
    top_p: Optional[float] = config.top_p
    max_tokens: Optional[int] = config.max_tokens
    presence_penalty: Optional[float] = config.presence_penalty
    frequency_penalty: Optional[float] = config.frequency_penalty


class BatchPredictionRequest(BaseModel):
    prompts: List[str]
    temperature: Optional[float] = config.temperature
    top_p: Optional[float] = config.top_p
    max_tokens: Optional[int] = config.max_tokens
    presence_penalty: Optional[float] = config.presence_penalty
    frequency_penalty: Optional[float] = config.frequency_penalty


# 定义API端点
@app.get("/")
async def root():
    return {"message": "Welcome to the Web3.0 AI Model API"}


@app.post("/predict")
async def predict(request: PredictionRequest):
    """单条预测"""
    try:
        # 创建采样参数
        sampling_params = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty
        }

        result = model_inference.predict(request.prompt, sampling_params)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict")
async def batch_predict(request: BatchPredictionRequest):
    """批量预测"""
    try:
        # 创建采样参数
        sampling_params = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty
        }

        results = model_inference.batch_predict(request.prompts, sampling_params)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)