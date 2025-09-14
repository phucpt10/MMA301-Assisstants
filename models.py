import os
import json
import time
import requests
from typing import Optional
import google.generativeai as genai

class LLMProvider:
    def __init__(self, provider: str):
        self.provider = provider.lower().strip()

        # Các tham số có thể override qua Secrets
        self.max_tokens = int(os.getenv("GITHUB_MODELS_MAX_TOKENS", "600"))
        self.max_context_chars = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
        # Timeout: tuple(connect, read). Có thể override bằng GITHUB_MODELS_TIMEOUT_READ/CONNECT
        connect_to = float(os.getenv("GITHUB_MODELS_TIMEOUT_CONNECT", "10"))
        read_to = float(os.getenv("GITHUB_MODELS_TIMEOUT_READ", "180"))
        self.timeout = (connect_to, read_to)
        # Số lần retry khi timeout/5xx
        self.retries = int(os.getenv("GITHUB_MODELS_RETRIES", "3"))
        self.backoff_base = float(os.getenv("GITHUB_MODELS_BACKOFF_BASE", "2"))

        if self.provider == "github":
            self.gh_token = os.getenv("GITHUB_MODELS_TOKEN")
            self.gh_model = os.getenv("GITHUB_MODELS_MODEL", "gpt-4o-mini")
            self.gh_base = os.getenv("GITHUB_MODELS_BASE_URL", "https://models.inference.ai.azure.com")
            if not self.gh_token:
                raise ValueError("Missing GITHUB_MODELS_TOKEN in secrets.")
        elif self.provider == "google":
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

    # ---------- GitHub Models ----------
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
        snippet = (body[:1000] + "...") if len(body) > 1000 else body
        return f"HTTP {resp.status_code} {resp.reason} | Body: {snippet}"

    def _post_with_retries(self, url: str, payload: dict) -> requests.Response:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = requests.post(
                    url,
                    headers=self._github_headers(),
                    data=json.dumps(payload),
                    timeout=self.timeout,
                )
                # Retry khi gặp 5xx
                if resp.status_code >= 500:
                    last_exc = RuntimeError(self._summarize_http_error(resp))
                    raise last_exc
                return resp
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError) as e:
                last_exc = e
                if attempt < self.retries:
                    time.sleep(self.backoff_base ** (attempt - 1))  # 1s, 2s, 4s...
                else:
                    raise
        # Fallback (hiếm khi tới đây)
        if isinstance(last_exc, Exception):
            raise last_exc
        raise RuntimeError("Unknown error while calling GitHub Models")

    def _generate_github(self, question: str, context: str, system_prompt: str, temperature: float) -> str:
        url = f"{self.gh_base}/chat/completions"

        # Cắt context quá dài để tránh thời gian suy luận quá lâu
        ctx = context or ""
        truncated = False
        if self.max_context_chars > 0 and len(ctx) > self.max_context_chars:
            ctx = ctx[: self.max_context_chars]
            truncated = True

        messages = [{"role": "system", "content": system_prompt}]
        if ctx:
            extra = "\n\n[Context truncated]\n" if truncated else ""
            messages.append({"role": "system", "content": f"Use this course context when relevant:{extra}\n{ctx}"})
        messages.append({"role": "user", "content": question})

        payload = {
            "model": self.gh_model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(self.max_tokens),
        }

        resp = self._post_with_retries(url, payload)
        if not resp.ok:
            raise RuntimeError(f"GitHub Models API error: {self._summarize_http_error(resp)}")
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    # ---------- Google (Gemini) ----------
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

    # Ping để test nhanh kết nối
    def ping(self) -> str:
        if self.provider == "github":
            url = f"{self.gh_base}/chat/completions"
            payload = {
                "model": self.gh_model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 5,
            }
            try:
                resp = requests.post(url, headers=self._github_headers(), data=json.dumps(payload), timeout=self.timeout)
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
