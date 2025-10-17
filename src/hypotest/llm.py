from dataclasses import dataclass
import subprocess, shutil

@dataclass
class LLMConfig:
    llm_provider: str = "ollama"
    model: str = "mistral"
    max_tokens: int = 400
    temperature: float = 0.2

class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    @classmethod
    def from_config(cls, cfg_dict: dict) -> "LLMClient":
        return cls(LLMConfig(**{**LLMConfig().__dict__, **cfg_dict}))

    def generate(self, prompt: str) -> str:
        if self.cfg.llm_provider == "ollama":
            if shutil.which("ollama") is None:
                raise RuntimeError("Ollama not found. Install with: brew install ollama")
            cmd = ["ollama", "run", self.cfg.model]
            res = subprocess.run(cmd, input=prompt.encode(), capture_output=True)
            if res.returncode != 0:
                raise RuntimeError(res.stderr.decode() or "Ollama failed")
            return res.stdout.decode()
        elif self.cfg.llm_provider == "rule_based":
            return "RULE_BASED_RESPONSE"
        else:
            raise NotImplementedError(f"Provider {self.cfg.llm_provider} not implemented")
