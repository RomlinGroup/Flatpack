# config.py
from pathlib import Path


class Config:
    HOME_DIR = Path.home()
    CONFIG_FILE_PATH = HOME_DIR / ".fpk_config.toml"
    GITHUB_REPO_URL = "https://api.github.com/repos/romlingroup/flatpack-ai"
    BASE_URL = "https://raw.githubusercontent.com/romlingroup/flatpack-ai/main/warehouse"
    LOGGING_ENDPOINT = "https://fpk.ai/api/index.php"

    def __init__(self):
        self.api_key = None
