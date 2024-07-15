import logging
from logging.handlers import TimedRotatingFileHandler
import signal
from concurrent import futures
import time
from main import Uie_Predictor
import argparse
import os
import sys
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.insert(0, parent_path)
from config import Config

Config = Config()

if not os.path.exists(os.path.dirname(Config.uie_log_file_path)):
    os.makedirs(os.path.dirname(Config.uie_log_file_path))



handler = TimedRotatingFileHandler(Config.uie_log_file_path, when="midnight", backupCount=Config.backupCount)
handler.setFormatter(logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'))
logger.addHandler(handler)

class UIEModel():
    def __init__(self):
        self.UP_model = Uie_Predictor(Config.up_model_path)
        self.author_model = Uie_Predictor(Config.author_model_path)
        self.identity_model = Uie_Predictor(Config.identity_model_path)
        self.abbreviation_model = Uie_Predictor(Config.abbreviation_model_path)

    def use_model(self, content, model_name):
        st = time.time()
        messages_data = eval(content)
        content_data = messages_data.get('content')
        entity_temp = messages_data.get('entities')
        state_code = 200
        if model_name.lower().strip() == Config.model_name_list[0]:
            if entity_temp:
                self.UP_model.set_schema(entity_temp)
            output = self.UP_model.model_pre(content_data)
            return {'result': output, 'costTime': time.time() - st, 'message': "success", 'state_code': state_code}
        elif model_name.lower().strip() == Config.model_name_list[1]:
            if entity_temp:
                self.author_model.set_schema(entity_temp)
            output = self.author_model.model_pre(content_data)
            return {'result': output, 'costTime': time.time() - st, 'message': "success", 'state_code': state_code}
        elif model_name.lower().strip() == Config.model_name_list[2]:
            if entity_temp:
                self.identity_model.set_schema(entity_temp)
            output = self.identity_model.model_pre(content_data)
            return {'result': output, 'costTime': time.time() - st, 'message': "success", 'state_code': state_code}
        elif model_name.lower().strip() == Config.model_name_list[3]:
            if entity_temp:
                self.abbreviation_model.set_schema(entity_temp)
            output = self.abbreviation_model.model_pre(content_data)
            return {'result': output, 'costTime': time.time() - st, 'message': "success", 'state_code': state_code}
        else:
            state_code = 500
            return {'result': '', 'costTime': 0.0,
                    'message': f'model_name is err ,must in {Config.model_name_list}',
                    'state_code': state_code}

UIE_Model = UIEModel()

app = FastAPI()

class HealthResponse(BaseModel):
    message: str

class PredictRequest(BaseModel):
    content: str
    model_name: str

class PredictResponse(BaseModel):
    result: dict

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    HTTP method that returns the health status of the backend service.
    """
    return {"message": "OK"}

@app.post("/uie", response_model=PredictResponse)
async def uie_predict(request: PredictRequest):
    try:
        model_name = request.model_name
        logger.info('模型名称，：{}'.format(model_name))
        logger.info('<<<模型入参>>>：{}'.format(request.content))
        response = UIE_Model.use_model(request.content, model_name)
        logger.info('<<<模型返回>>：{}'.format(response))
        return {"result": response}
    except Exception as e:
        logger.error('模型处理出错，：{}'.format(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=Config.uie_port)
