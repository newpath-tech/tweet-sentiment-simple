"""Prepare reviewed TweetClaw exports for the Streamlit dashboard."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


FIELDNAMES = ("tweet", "source_id", "created_at", "author_username", "url")

FIELD_PATHS = {
    "tweet": (
        ("tweet",),
        ("text",),
        ("full_text",),
        ("fullText",),
        ("content",),
        ("body",),
        ("rawContent",),
        ("data", "text"),
        ("tweet", "text"),
        ("legacy", "full_text"),
    ),
    "source_id": (
        ("id",),
        ("tweet_id",),
        ("tweetId",),
        ("rest_id",),
        ("data", "id"),
        ("tweet", "id"),
    ),
    "created_at": (
        ("created_at",),
        ("createdAt",),
        ("date",),
        ("timestamp",),
        ("data", "created_at"),
        ("tweet", "created_at"),
    ),
    "author_username": (
        ("author_username",),
        ("username",),
        ("screen_name",),
        ("author", "username"),
        ("author", "screen_name"),
        ("user", "username"),
        ("user", "screen_name"),
        ("tweet", "author", "username"),
        ("tweet", "author", "screen_name"),
    ),
    "url": (
        ("url",),
        ("tweet_url",),
        ("tweetUrl",),
        ("data", "url"),
        ("tweet", "url"),
    ),
}


def value_at_path(record: Mapping[str, Any], path: tuple[str, ...]) -> str:
    current: Any = record
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return ""
        current = current[key]
    if current is None or isinstance(current, (Mapping, list)):
        return ""
    return str(current).strip()


def first_value(record: Mapping[str, Any], field: str) -> str:
    for path in FIELD_PATHS[field]:
        value = value_at_path(record, path)
        if value:
            return value
    return ""


def normalize_record(record: Mapping[str, Any]) -> dict[str, str] | None:
    tweet = " ".join(first_value(record, "tweet").split())
    if not tweet:
        return None
    return {
        "tweet": tweet,
        "source_id": first_value(record, "source_id"),
        "created_at": first_value(record, "created_at"),
        "author_username": first_value(record, "author_username"),
        "url": first_value(record, "url"),
    }


def records_from_csv(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        yield from csv.DictReader(handle)


def records_from_json_data(data: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(data, list):
        for item in data:
            if isinstance(item, Mapping):
                yield item
        return

    if isinstance(data, Mapping):
        for key in ("tweets", "data", "items", "results"):
            value = data.get(key)
            if isinstance(value, list):
                yield from records_from_json_data(value)
                return
        yield data


def records_from_json_or_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL on line {line_number}: {error}") from error
            if isinstance(item, Mapping):
                yield item
        return

    yield from records_from_json_data(data)


def load_records(path: Path) -> Iterable[Mapping[str, Any]]:
    if path.suffix.lower() == ".csv":
        yield from records_from_csv(path)
        return
    yield from records_from_json_or_jsonl(path)


def prepare_rows(input_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for record in load_records(input_path):
        row = normalize_record(record)
        if row is None:
            continue
        dedupe_key = row["source_id"] or row["tweet"].casefold()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_text(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(row["tweet"] for row in rows) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert reviewed TweetClaw JSON, JSONL, or CSV exports into dashboard input files."
    )
    parser.add_argument("input", type=Path, help="Path to the TweetClaw export.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/tweets.csv"),
        help="CSV output path. Defaults to data/tweets.csv.",
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        help="Optional newline-separated text output for quick paste tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = prepare_rows(args.input)
    if not rows:
        raise SystemExit("No tweet text found in the input export.")
    write_csv(rows, args.output)
    if args.text_output is not None:
        write_text(rows, args.text_output)


if __name__ == "__main__":
    main()
