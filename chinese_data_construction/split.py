import re
from typing import Optional

from utils import logger

count_split_success = 0
count_split_failed = 0

CHINESE_NUM_PATTERN = r"0-9零〇一二三四五六七八九十百千万两上下前后终初末廿卅"
CHAPTER_TOKEN = rf"第[{CHINESE_NUM_PATTERN}]+[章节回卷部篇集册幕]"
SPECIAL_TITLES = r"序章|序言|楔子|引子|前言|后记|尾声|终章|番外(?:篇)?|外传|附录"
ENGLISH_CHAPTER_RE = re.compile(
    r"^(?:#{1,6}\s+.*|(?:Chapter|CHAPTER|Prologue|Epilogue|Afterword|Preface|Introduction|Conclusion|Appendix|Interlude|Part|Book)\s+.+)$",
    re.IGNORECASE,
)
CHINESE_CHAPTER_RE = re.compile(
    rf"^(?:{CHAPTER_TOKEN}(?:\s+.*)?|卷[{CHINESE_NUM_PATTERN}]+(?:\s+.*)?|{SPECIAL_TITLES}(?:\s+.*)?|"
    rf"(?:[☆★◆◇●○◎]\s*[、.．]?\s*)?(?:{CHAPTER_TOKEN}|{SPECIAL_TITLES}|[【\[]?[一二三四五六七八九十百千万零两0-9]+[】\]])(?:\s+.*)?)$"
)
NUMBERED_HEADING_RE = re.compile(
    r"^(?:0*\d{1,4}(?:、|．|\.(?!\d))|[一二三四五六七八九十百千万零两]{1,8}[、.．])\s*\S+.*$"
)
SPACED_NUM_HEADING_RE = re.compile(r"^(?:0*\d{1,4}|[一二三四五六七八九十百千万零两]{1,8})\s+\S+.*$")
PAREN_SERIES_HEADING_RE = re.compile(
    r"^[^\s]{1,30}(?:\s+[^\s]{1,20}){0,2}\s*[（(][0-9０-９一二三四五六七八九十百千万零两]+[）)]$"
)
DAY_SCENE_HEADING_RE = re.compile(
    rf"^第[{CHINESE_NUM_PATTERN}]+(?:天|场)(?:\s+.*)?$"
)
PART_HEADING_RE = re.compile(
    rf"^第[{CHINESE_NUM_PATTERN}]+(?:部分?|部)(?:[-－]\d+)?$"
)
CUSTOM_BOOK_SHORT_TITLES = {
    "零的焦点": {
        "丈夫",
        "失踪",
        "北方的疑惑",
        "地方名士",
        "沿海的坟场",
        "大伯子的行动",
        "前历",
        "毒死者",
        "北陆铁道",
        "逃亡",
        "丈夫的意义",
        "雪国的不安",
    },
}


def normalize_line_for_title(line: str) -> str:
    return re.sub(r"\s+", " ", line.replace("\u3000", " ")).strip()


def normalize_book_title(book_title: Optional[str]) -> str:
    if not book_title:
        return ""
    return re.sub(r"-未知作者$", "", book_title.strip())


def is_directory_line(line: str) -> bool:
    stripped = line.strip()
    return stripped == "目录" or bool(re.search(r"[.．·•…]{4,}", stripped))


def classify_chapter_title(line: str, book_title: Optional[str] = None) -> Optional[str]:
    normalized_title = normalize_book_title(book_title)
    if ENGLISH_CHAPTER_RE.match(line) or CHINESE_CHAPTER_RE.match(line):
        return "standard"
    if DAY_SCENE_HEADING_RE.match(line):
        return "day_scene"
    if PART_HEADING_RE.match(line):
        return "part"
    if PAREN_SERIES_HEADING_RE.match(line):
        return "paren_series"
    if NUMBERED_HEADING_RE.match(line):
        return "numbered"
    if SPACED_NUM_HEADING_RE.match(line):
        return "spaced_numbered"
    if line in CUSTOM_BOOK_SHORT_TITLES.get(normalized_title, set()):
        return "book_specific"
    return None


def looks_like_chapter_title(line: str, book_title: Optional[str] = None) -> bool:
    stripped = normalize_line_for_title(line)
    if not stripped or len(stripped) > 80:
        return False
    if stripped in {"正文", "目录"} or is_directory_line(stripped):
        return False
    kind = classify_chapter_title(stripped, book_title)
    if kind in {"numbered", "spaced_numbered"}:
        return len(stripped) <= 35
    return kind is not None


def find_chapter_matches(content: str, book_title: Optional[str] = None):
    lines = content.splitlines(keepends=True)
    offset = 0
    matches = []
    for idx, raw_line in enumerate(lines):
        line_without_newline = raw_line.rstrip("\n")
        stripped = normalize_line_for_title(line_without_newline)
        kind = classify_chapter_title(stripped, book_title)
        prev_blank = idx == 0 or not lines[idx - 1].strip()
        next_blank = idx + 1 >= len(lines) or not lines[idx + 1].strip()
        weak_title = kind in {"numbered", "spaced_numbered", "paren_series", "book_specific"}
        if kind and (not weak_title or prev_blank or next_blank):
            matches.append(
                {
                    "start": offset,
                    "title": line_without_newline.strip(),
                }
            )
        offset += len(raw_line)
    return matches


def merge_short_splits(chapter_splits, min_chars=2000):
    merged = []
    chunk = {"title": None, "content": ""}

    for split in chapter_splits:
        if not split["content"].strip():
            continue
        if chunk["title"] is None:
            chunk["title"] = split["title"]
        chunk["content"] += ("" if not chunk["content"] else "\n") + split["content"].strip()
        if len(chunk["content"]) >= min_chars:
            merged.append(chunk)
            chunk = {"title": None, "content": ""}

    if chunk["content"]:
        merged.append(chunk)
    return merged


def split_is_valid(original_content: str, chapter_splits) -> bool:
    titled_splits = [split for split in chapter_splits if split.get("title")]
    if len(titled_splits) >= 5:
        return True
    if len(titled_splits) >= 3:
        covered_chars = sum(len(split["content"]) for split in titled_splits)
        return covered_chars / max(len(original_content), 1) >= 0.35
    return False


def split_book(book: dict) -> dict:
    """
    Split the book content into chapter-aware chunks.
    """
    content = book["content"]
    chapters = find_chapter_matches(content, book.get("title"))

    chapter_splits = []
    for i, match in enumerate(chapters):
        start = match["start"]
        end = chapters[i + 1]["start"] if i + 1 < len(chapters) else len(content)

        if i == 0:
            prefatory_content = content[:start].strip()
            if prefatory_content:
                chapter_splits.append({"title": None, "content": prefatory_content})

        chapter_content = content[start:end].strip()
        if chapter_content:
            chapter_splits.append({"title": match["title"], "content": chapter_content})

    chapter_splits = merge_short_splits(chapter_splits)
    logger.info(
        f'Splitting {book["title"]} ({book.get("num_tokens", -1)} tokens) into {len(chapter_splits)} chapter chunks'
    )

    if split_is_valid(content, chapter_splits):
        global count_split_success
        count_split_success += 1
        return chapter_splits

    global count_split_failed
    count_split_failed += 1
    return None


if __name__ == "__main__":
    try:
        import jsonlines
        with jsonlines.open("data/src/books_example.jsonl", mode="r") as reader:
            books_data = list(reader)
    except ImportError:
        import json
        with open("data/src/books_example.jsonl", "r", encoding="utf-8") as f:
            books_data = [json.loads(line) for line in f if line.strip()]

    split_books = [split_book(book) for book in books_data]
    print(f"Split {count_split_success} books, failed {count_split_failed} books")
    books_data = split_books
    print(f"Processed {len(books_data)} books, splitting their content into chapters.")