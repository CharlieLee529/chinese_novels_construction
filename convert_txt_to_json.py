from __future__ import annotations

import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data"


def parse_title_author(stem: str) -> tuple[str, str]:
    title, separator, author = stem.rpartition("-")
    if separator:
        return title, author
    return stem, ""


def convert_subdirectory(subdir: Path) -> tuple[int, int]:
    records: list[dict[str, str]] = []
    missing_author_count = 0

    for txt_path in sorted(subdir.rglob("*.txt")):
        title, author = parse_title_author(txt_path.stem)
        if not author:
            missing_author_count += 1

        record = {
            "title": title,
            "author": author,
            "content": txt_path.read_text(encoding="utf-8", errors="replace"),
        }

        json_path = txt_path.with_suffix(".json")
        json_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        records.append(record)

    jsonl_path = subdir / f"{subdir.name}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")

    return len(records), missing_author_count


def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    total_files = 0
    total_missing_author = 0

    for subdir in sorted(path for path in DATA_DIR.iterdir() if path.is_dir()):
        file_count, missing_author_count = convert_subdirectory(subdir)
        total_files += file_count
        total_missing_author += missing_author_count
        print(
            f"{subdir.name}: converted {file_count} files, "
            f"missing author in {missing_author_count} files"
        )

    print(
        f"Done. Converted {total_files} txt files across "
        f"{len([p for p in DATA_DIR.iterdir() if p.is_dir()])} subdirectories. "
        f"Missing author in {total_missing_author} files."
    )


if __name__ == "__main__":
    main()
