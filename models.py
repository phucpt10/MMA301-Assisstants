import os
import json
import requests
from typing import Optional
import google.generativeai as genai

class LLMProvider:
    def __init__(self, provider: str):
        self.provider = provider
        if provider == "github":
            self.gh_token = os.getenv("GITHUB_MODELS_TOKEN")
            self.gh_model = os.getenv("GITHUB_MODELS_MODEL", "gpt-4o-mini")
            self.gh_base = os.getenv("GITHUB_MODELS_BASE_URL", "https://models.inference.ai.azure.com")
            if not self.gh_token:
                raise ValueError("Missing GITHUB_MODELS_TOKEN in secrets.")
        elif provider == "google":
            self.gg_key = os.getenv("GOOGLE_API_KEY")
            self.gg_model = os.getenv("GOOGLE_MODEL", "gemini-1.5-pro")
            if not self.gg_key:
                raise ValueError("Missing GOOGLE_API_KEY in secrets.")
            genai.configure(api_key=self.gg_key)
            self.gg_client = genai.GenerativeModel(self.gg_model)
        else:
            raise ValueError("Provider must be 'github' or 'google'.")

    @staticmethod
    def from_env():
        provider = os.getenv("PROVIDER", "github").lower()
        return LLMProvider(provider)

    def generate_answer(self, question: str, context: str, system_prompt: str, temperature: float = 0.3) -> str:
        if self.provider == "github":
            return self._generate_github(question, context, system_prompt, temperature)
        return self._generate_google(question, context, system_prompt, temperature)

    def _generate_github(self, question: str, context: str, system_prompt: str, temperature: float) -> str:
        url = f"{self.gh_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.gh_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        messages = [{"role": "system", "content": system_prompt}]
        if context:
            messages.append({"role": "system", "content": f"Use this course context when relevant:\n{context}"})
        messages.append({"role": "user", "content": question})
        payload = {
            "model": self.gh_model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": 800
        }
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _generate_google(self, question: str, context: str, system_prompt: str, temperature: float) -> str:
        prompt = self._build_prompt(system_prompt, question, context)
        resp = self.gg_client.generate_content(
            prompt,
            generation_config={"temperature": float(temperature)}
        )
        return getattr(resp, "text", "").strip() or "Xin lỗi, tôi không thể tạo câu trả lời lúc này."

    def _build_prompt(self, system_prompt: str, question: str, context: str) -> str:
        ctx = f"\n\nContext (course/vendor docs):\n{context}\n" if context else ""
        return f"{system_prompt}{ctx}\nUser question:\n{question}\n"
