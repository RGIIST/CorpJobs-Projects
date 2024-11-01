from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.router import router as DefaultRouter

description = """
Passport Extraction Model
"""
tags_metadata = [
    {
        "name": "passport Model",
        "description": "python side code for passport extraction",
        "externalDocs": {
            "description": "Api description",
            "url": "https://fastapi.tiangolo.com/",
        },
    },
]


app = FastAPI(
    title="Passports",
    description=description,
    version="0.0.1",
    contact={
        "name": "RaviGautam",
        "email": "ravi.gautam@dexninja.com",
    },
    openapi_tags=tags_metadata,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["content-disposition"]
)

app.include_router(DefaultRouter, tags=["default"], prefix="/api")
