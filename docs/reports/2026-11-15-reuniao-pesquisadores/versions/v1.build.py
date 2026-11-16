"""Build docs/reports/2026-11-15-reuniao-pesquisadores.html from page.template.html and extracts/*.json.

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
    findings = {
        key: [{k: f[k] for k in ("experiments", "verdict", "key_numbers")} for f in s["findings"]]
        for key, s in sections.items()
    }
    template = (HERE / "page.template.html").read_text(encoding="utf-8")
    wanted = set(re.findall(r'data-table="([^"]+)"', template)) | set(re.findall(r'T\("([^"]+)"\)', template))
    missing = sorted(wanted - tables.keys())
    assert not missing, f"tables the page asks for but no extract holds: {missing}"
    built = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y-%m-%d %H:%M GMT-3")
    data = json.dumps({"tables": tables, "findings": findings}, ensure_ascii=False).replace("</", "<\\/")
    OUT.write_text(template.replace("__BUILT__", built).replace("__DATA__", data), encoding="utf-8")
    print(f"{OUT} · {len(tables)} tables · {sum(map(len, findings.values()))} findings · built {built}")


if __name__ == "__main__":
    main()
