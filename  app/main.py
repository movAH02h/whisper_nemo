from fastapi import fastAPI
from api.endpoints import router as api_router

app = FastAPI(title=settings.APP_NAME)

app.include_router(api_router, prefix="/api/v1")

frontend_path = os.path.join(os.path.dirname(__file__), "../frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

@app.on_event("startup")
async def startup_event():
    os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
