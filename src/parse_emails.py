"""Parse raw SpamAssassin corpus emails into a single structured parquet file."""

import email
import email.policy
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "emails.parquet"

# folder name -> label
FOLDERS = {
    "easy_ham": "ham",
    "easy_ham_2": "ham",
    "hard_ham": "ham",
    "spam": "spam",
    "spam_2": "spam",
}


def extract_body(msg: email.message.EmailMessage) -> str:
    """Return plain-text body, falling back to a stripped text/html part."""
    if msg.is_multipart():
        plain_part = None
        html_part = None
        for part in msg.walk():
            if part.is_multipart():
                continue
            content_type = part.get_content_type()
            if content_type == "text/plain" and plain_part is None:
                plain_part = part
            elif content_type == "text/html" and html_part is None:
                html_part = part
        target = plain_part or html_part
        if target is None:
            return ""
    else:
        target = msg

    try:
        content = target.get_content()
    except (LookupError, UnicodeDecodeError, Exception):
        payload = target.get_payload(decode=True)
        if not payload:
            content = ""
        else:
            charset = target.get_content_charset() or "utf-8"
            try:
                content = payload.decode(charset, errors="replace")
            except LookupError:
                content = payload.decode("utf-8", errors="replace")

    if target.get_content_type() == "text/html":
        content = strip_html(content)

    return content


def strip_html(html: str) -> str:
    """Very lightweight tag stripper (no external deps for this pass)."""
    import re

    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_file(path: Path, label: str, source_folder: str) -> dict | None:
    with open(path, "rb") as f:
        raw_bytes = f.read()

    msg = email.message_from_bytes(raw_bytes, policy=email.policy.default)

    received_count = len(msg.get_all("Received", []))

    return {
        "label": label,
        "source_folder": source_folder,
        "filename": path.name,
        "from": msg.get("From", ""),
        "to": msg.get("To", ""),
        "subject": msg.get("Subject", ""),
        "date": msg.get("Date", ""),
        "reply_to": msg.get("Reply-To", ""),
        "received_count": received_count,
        "body": extract_body(msg),
    }


def main():
    rows = []
    skipped = 0
    skip_log = []

    for folder_name, label in FOLDERS.items():
        folder_path = RAW_DIR / folder_name
        if not folder_path.exists():
            print(f"WARNING: folder not found, skipping: {folder_path}")
            continue

        files = [p for p in folder_path.iterdir() if p.is_file()]
        print(f"Parsing {folder_name} ({len(files)} files, label={label})...")

        for file_path in files:
            try:
                row = parse_file(file_path, label, folder_name)
                rows.append(row)
            except Exception as e:
                skipped += 1
                skip_log.append((str(file_path), str(e)))

    df = pd.DataFrame(rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    print(f"\nParsed {len(df)} emails -> {OUT_PATH}")
    print(f"Skipped {skipped} files due to parse errors")
    if skip_log:
        print("First few skip reasons:")
        for path, err in skip_log[:5]:
            print(f"  {path}: {err}")

    print("\nLabel counts:")
    print(df["label"].value_counts())
    print("\nSource folder counts:")
    print(df["source_folder"].value_counts())


if __name__ == "__main__":
    main()
