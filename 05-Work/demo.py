import os
import json
import time
import loguru
from pydantic import BaseModel
import sys
from paddlenlp import Taskflow
from fastapi import FastAPI, HTTPException

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_path)


class Uie_Predictor:
    def __init__(self, model_path):
        self.model = Taskflow("information_extraction", schema=["人物"], task_path=model_path)
    
    def set_schema(self, entity_list):
        self.model.set_schema(entity_list)
    
    def model_pre(self, text):
        return self.model(text)

class UIEModel():
    