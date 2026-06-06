from __future__ import annotations

import argparse
import html
import subprocess
import tempfile
from pathlib import Path

import markdown


CSS = """
body {
  font-family: "Noto Sans CJK SC", "Microsoft YaHei", "WenQuanYi Micro Hei", Arial, sans-serif;
  color: #111827;
  line-height: 1.62;
  max-width: 980px;
  margin: 0 auto;
  padding: 28px 36px;
}
h1 {
  font-size: 28px;
  border-bottom: 2px solid #1f2937;
  padding-bottom: 10px;
}
h2 {
  font-size: 21px;
  margin-top: 30px;
  border-left: 5px solid #2563eb;
  padding-left: 10px;
}
h3 {
  font-size: 17px;
}
table {
  width: 100%;
  border-collapse: collapse;
  margin: 14px 0 20px;
  font-size: 13px;
}
th, td {
  border: 1px solid #d1d5db;
  padding: 7px 8px;
  vertical-align: top;
}
th {
  background: #f3f4f6;
  font-weight: 700;
}
code {
  font-family: "DejaVu Sans Mono", Consolas, monospace;
  background: #f3f4f6;
  padding: 1px 4px;
  border-radius: 3px;
  font-size: 12px;
}
pre {
  background: #111827;
  color: #f9fafb;
  padding: 12px;
  border-radius: 6px;
  overflow-wrap: break-word;
  white-space: pre-wrap;
  font-size: 12px;
}
pre code {
  background: transparent;
  color: inherit;
  padding: 0;
}
img {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 12px auto 22px;
  border: 1px solid #e5e7eb;
}
p {
  margin: 9px 0;
}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_pdf = args.output.resolve() if args.output else input_path.with_suffix(".pdf")
    output_html = output_pdf.with_suffix(".html")

    text = input_path.read_text(encoding="utf-8")
    body = markdown.markdown(
        text,
        extensions=["extra", "tables", "fenced_code", "toc", "sane_lists"],
        output_format="html5",
    )
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{html.escape(input_path.stem)}</title>
  <style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""
    output_html.write_text(document, encoding="utf-8")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    user_installation = Path(tempfile.mkdtemp(prefix="lo_markdown_pdf_"))
    subprocess.run(
        [
            "libreoffice",
            "--headless",
            f"-env:UserInstallation=file://{user_installation}",
            "--convert-to",
            "pdf",
            "--outdir",
            str(output_pdf.parent),
            str(output_html),
        ],
        cwd=str(input_path.parent),
        check=True,
    )
    converted = output_pdf.parent / output_html.with_suffix(".pdf").name
    if converted != output_pdf and converted.exists():
        converted.replace(output_pdf)
    print(output_pdf)


if __name__ == "__main__":
    main()
