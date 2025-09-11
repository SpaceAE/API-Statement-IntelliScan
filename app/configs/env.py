from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvSettings(BaseSettings):
	message: str = '🎉'

	model_config = SettingsConfigDict(env_file='.env')


env = EnvSettings()
