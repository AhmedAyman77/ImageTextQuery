from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Config settings for the application
    
    defines the configuration settings for the application using Pydantic.
    For example:
    - DATABASE_URL: str
    - SECRET_KEY: str
    """
    QDRANT_HOST: str
    QDRANT_API_KEY: str

    SQL_PORT: int
    SQL_HOST: str
    SQL_USER: str
    SQL_PASSWORD: str
    SQL_DATABASE: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


def get_settings_object():
    """
    Get the settings object
    
    This function creates an instance of the Settings class, which loads
    configuration settings from environment variables or a .env file.
    
    Returns:
        Settings: An instance of the Settings class with loaded configuration settings.
    """
    
    return Settings()