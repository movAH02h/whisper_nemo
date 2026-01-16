from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Speech Transcription API"
    UPLOAD_FOLDER: str = "temp_uploads"
    MAX_FILE_SIZE: int = 100 * 1024 * 1024
    ALLOWED_EXTENSIONS: set = {".wav", ".mp3", ".ogg", ".flac"}

    HF_TOKEN: str = ""

    class Config:
        env_file = ".env"