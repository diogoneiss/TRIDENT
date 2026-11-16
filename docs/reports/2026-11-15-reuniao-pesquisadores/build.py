"""Build docs/reports/2026-11-15-reuniao-pesquisadores.html (page) and -slides.html (deck) from their templates and extracts/*.json.

The extracts copy ticket Outcomes, ADRs and the experimentation log verbatim, each cell with its
path:line; the page renders every table and chart from them, so no number is typed into the HTML
by hand. Run: python3 docs/reports/2026-11-15-reuniao-pesquisadores/build.py
"""

import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / f"{HERE.name}.html"


def main() -> None:
    sections = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((HERE / "extracts").glob("*.json"))}
    tables: dict[str, dict] = {}
    for key, section in sections.items():
        for table in section["tables"]:
            assert table["id"] not in tables, f"duplicate table id {table['id']}"
            tables[table["id"]] = {**table, "section": key}
    for name, out in (("page.template.html", OUT), ("slides.template.html", HERE.parent / f"{HERE.name}-slides.html")):
        render(tables, (HERE / name).read_text(encoding="utf-8"), out)


def render(tables: dict[str, dict], template: str, out: Path) -> None:
    wanted = set(re.findall(r'data-table="([^"]+)"', template)) | set(re.findall(r'T\("([^"]+)"\)', template))
    missing = sorted(wanted - tables.keys())
    assert not missing, f"tables the page asks for but no extract holds: {missing}"
    built = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y-%m-%d %H:%M GMT-3")
    data = json.dumps({"tables": {k: tables[k] for k in sorted(wanted)}}, ensure_ascii=False).replace("</", "<\\/")
    out.write_text(template.replace("__BUILT__", built).replace("__DATA__", data), encoding="utf-8")
    print(f"{out} · {len(wanted)} of {len(tables)} tables embedded · built {built}")


if __name__ == "__main__":
    main()
