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

    def _github_headers(self):
        return {
            "Authorization": f"Bearer {self.gh_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _summarize_http_error(self, resp: requests.Response) -> str:
        body = ""
        try:
            body = resp.text or ""
        except Exception:
            body = ""
        # cắt ngắn body để tránh log quá dài
        body_snippet = (body[:1000] + "...") if len(body) > 1000 else body
        return f"HTTP {resp.status_code} {resp.reason} | Body: {body_snippet}"

    def _generate_github(self, question: str, context: str, system_prompt: str, temperature: float) -> str:
        url = f"{self.gh_base}/chat/completions"
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
        resp = requests.post(url, headers=self._github_headers(), data=json.dumps(payload), timeout=60)
        if not resp.ok:
            # Ném lỗi kèm tóm tắt chi tiết để bạn thấy rõ trong Logs
            raise RuntimeError(f"GitHub Models API error: {self._summarize_http_error(resp)}")
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

    # Thêm hàm ping để test nhanh kết nối
    def ping(self) -> str:
        if self.provider == "github":
            url = f"{self.gh_base}/chat/completions"
            payload = {
                "model": self.gh_model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 10,
            }
            try:
                resp = requests.post(url, headers=self._github_headers(), data=json.dumps(payload), timeout=30)
                if not resp.ok:
                    return f"❌ GitHub Models ping failed: {self._summarize_http_error(resp)}"
                return "✅ GitHub Models ping OK"
            except Exception as e:
                return f"❌ GitHub Models ping exception: {e}"
        else:
            try:
                r = self.gg_client.generate_content("ping")
                ok = bool(getattr(r, "text", "").strip())
                return "✅ Google (Gemini) ping OK" if ok else "❌ Google ping failed (empty response)"
            except Exception as e:
                return f"❌ Google ping exception: {e}"
