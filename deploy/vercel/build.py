"""Build script: assemble static frontend pages for the Vercel deploy.

Reads the shared shell plus per-page content files from ./src and
writes final pages into ./ (the Vercel root). Run after editing any
file under ./src:

    python build.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

PAGES = {
    "index.html":      {"title": "Dashboard",          "nav": "dashboard"},
    "dashboard.html":  {"title": "Analytics Dashboard", "nav": "dashboard"},
    "analysis.html":   {"title": "Analysis",           "nav": "analysis"},
    "history.html":    {"title": "Prediction History", "nav": "history"},
    "about.html":      {"title": "About",              "nav": "about"},
    "result.html":     {"title": "Analysis Result",    "nav": None},
}

NAV_MARKERS = {
    "dashboard": "<!--NAV_DASHBOARD-->",
    "analysis": "<!--NAV_ANALYSIS-->",
    "history": "<!--NAV_HISTORY-->",
    "about": "<!--NAV_ABOUT-->",
}


def main() -> None:
    shell = (SRC / "_shell.html").read_text(encoding="utf-8")
    for filename, meta in PAGES.items():
        content = (SRC / filename).read_text(encoding="utf-8")
        page = shell
        page = page.replace("<!--TITLE-->", meta["title"])
        for nav_key, marker in NAV_MARKERS.items():
            active = " active" if meta["nav"] == nav_key else ""
            page = page.replace(marker, active)
        page = page.replace("<!--CONTENT-->", content.rstrip())
        page = page.replace("<!--SCRIPTS-->", "")
        (ROOT / filename).write_text(page, encoding="utf-8")
        print(f"Built {filename}")


if __name__ == "__main__":
    main()