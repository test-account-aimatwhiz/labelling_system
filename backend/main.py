from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from backend.api.routes import router

app = FastAPI(title="SAM3 Labeling Tool")

# Routes
app.include_router(router)

# Static files
app.mount(
    "/static",
    StaticFiles(directory="frontend/static"),
    name="static"
)

# Templates
templates = Jinja2Templates(
    directory="frontend/templates"
)