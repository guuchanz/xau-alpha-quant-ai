import yaml
from pathlib import Path
from typing import Dict, Any

class Config:
    _instance: Dict[str, Any] = None

    @classmethod
    def load(cls, config_path: str = "configs/config.yaml") -> Dict[str, Any]:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"หาไฟล์ Config ไม่พบที่: {path.absolute()}")
        
        with open(path, "r", encoding="utf-8") as f:
            cls._instance = yaml.safe_load(f)
        return cls._instance

    @classmethod
    def get(cls) -> Dict[str, Any]:
        if cls._instance is None:
            return cls.load()
        return cls._instance