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
    title="SQL Server QA",
    description="API de QA sobre tablas en SQL Server en lenguage natural",
    version = "1.0",
    middleware=middleware
)

app.include_router(router)
