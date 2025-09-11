import yaml
import trafilatura

def load_vendor_urls(path: str = "sources.yaml"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        urls = data.get("urls", [])
        # Lọc trùng
        urls = list(dict.fromkeys([u.strip() for u in urls if isinstance(u, str) and u.strip()]))
        return urls
    except Exception:
        return []

def fetch_one(url: str) -> str:
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return ""
        text = trafilatura.extract(
            downloaded,
            include_formatting=False,
            include_links=False,
            include_images=False,
            include_tables=False
        )
        return (text or "").strip()
    except Exception:
        return ""

def fetch_vendor_docs(urls):
    docs = []
    for u in urls:
        text = fetch_one(u)
        if text:
            docs.append({"text": text, "source": u})
    return docs
