from fastapi import APIRouter, UploadFile
from app.services.transcription import TranscriptionService
import os

router = APIRouter()

@router.post("/transcibe")
async def transcribe_audio(file: UploadFile = File(...)):
    temp_path = f"temp_files/{file.filename}"
    with open(temp_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    try:
        service = TranscriptionService(setting.HF_TOKEN)
        result = service.process_audio(temp_path)

        os.remove(temp_path)
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPExceptioin(500, f"Ошибка обработки {str(e)}")