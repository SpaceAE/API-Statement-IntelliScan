from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
	model_config = SettingsConfigDict(
		env_file='.env',
		env_ignore_empty=True,
	)

	PROJECT_NAME: str = 'API Statement IntelliScan'

	ENVIRONMENT: Literal['development', 'production'] = 'development'
	API_PREFIX: str = '/api/v1'
	MODEL_PATH: str = 'model.h5'


settings = Settings()
