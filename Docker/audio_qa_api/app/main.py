from fastapi import FastAPI

from src.routers import router

from starlette.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

app = FastAPI(
    title="Media QA",
    description="API de QA sobre transcritos obtenidos de audios o videos",
    version = "1.0",
    middleware=middleware
)

app.include_router(router)