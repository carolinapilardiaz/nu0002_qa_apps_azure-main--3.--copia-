from typing import Optional, Dict, List

from pydantic import BaseModel, EmailStr, Field


class UserMessageModel(BaseModel):

    user_message: str
    user_id: str
    session_id: str
   
    class Config:
        
        schema_extra = {
            "example": {
                "user_message": "Hola, podrias presentarte?",
                "user_id": "usuario_test",
                "session_id": "00001"
                }
            }
        

class AIResponseModel(BaseModel):

    message: str


class ErrorMessageModel(BaseModel):

    error: str
    description: Optional[str]