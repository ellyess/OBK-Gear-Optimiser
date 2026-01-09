"""
Scrape equipment stats from ohbabykart.wiki.gg and export as PARTS_DATABASE.

This script fetches rendered HTML using the MediaWiki API (more reliable than
Playwright for wiki.gg), parses stat tables, normalizes stat keys, then formats
the output into a Python module and JSON file for an equipment builder app.

Outputs:
    - scraped_out/parts_database.py 
    - scraped_out/parts_database.json
"""

import json
import math
import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


PAGE = "Equipment_Stats"
BASE = "https://ohbabykart.wiki.gg"

API_CANDIDATES = [
    f"{BASE}/api.php",
    f"{BASE}/w/api.php",
]

# ----------------- postprocess config -----------------

# Stat renames to match app schema.
STAT_RENAMES = {
    "DazeMulti": "Daze",
    "UltChargeMulti": "UltCharge",
    "UltStartAmount": "UltStart",
    "SlipStreamTime": "SlipTime",
    "SlowAreaSpd": "SlowDownSpd",
}

# Stats to drop entirely.
DROP_STATS = {"Rarity"}

# Category detection from item name suffix.
CATEGORY_SUFFIXES = [
    (" Engine", "ENGINE"),
    (" Exhaust", "EXHAUST"),
    (" Suspension", "SUSPENSION"),
    (" Gearbox", "GEARBOX"),
    (" Trinket", "TRINKET"),
    (" Keys", "TRINKET"),
    (" Tags", "TRINKET"),
    (" Tag", "TRINKET"),
]

# Output order to my app.
DB_ORDER = ["ENGINE", "EXHAUST", "SUSPENSION", "GEARBOX", "TRINKET"]

# Stat ordering to match app. Unknown stats are appended alphabetically.
STAT_ORDER = [
    "BoostPads",
    "DriftRate",
    "T1",
    "T2",
    "T3",
    "StartBoost",
    "CoinBoostSpd",
    "CoinBoostTime",
    "StartCoins",
    "MaxCoins",
    "MaxCoinsSpd",
    "DriftSteer",
    "Steer",
    "AirDriftTime",
    "Speed",
    "SlowDownSpd",
    "Daze",
    "UltCharge",
    "UltStart",
    "SlipStreamRadius",
    "SlipStreamSpd",
    "SlipTime",
    "TrickSpd",
]


# ----------------- normalization -----------------
def norm_key(text):
    """Normalize a table header into a CamelCase-ish stat key.

    This strips footnotes, normalizes minus characters, removes punctuation,
    and converts multi-word strings into a single concatenated token.

    Args:
        text: Raw column header text from the wiki table.

    Returns:
        A normalized key string (e.g., "Slip Stream Spd" -> "SlipStreamSpd").
        Returns an empty string if the input does not produce a valid key.
    """
    s = str(text).strip()
    s = re.sub(r"\[\d+\]", "", s)          # remove footnotes like [1]
    s = s.replace("−", "-")               # normalize minus
    s = re.sub(r"[^0-9A-Za-z]+", " ", s)  # keep alnum, spaces
    parts = [p for p in s.split() if p]
    if not parts:
        return ""
    return parts[0] + "".join(p[:1].upper() + p[1:] for p in parts[1:])


def to_number(value):
    """Convert a cell value to float when possible.

    Handles:
        - blanks and em-dashes -> None
        - commas in numbers
        - percent strings like "50%" -> 50.0 (not 0.5)
        - weird minus sign "−"

    Args:
        value: Any cell value from pandas read_html.

    Returns:
        A float if parsable, otherwise None.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    t = str(value).strip()
    if not t or t in {"—", "-", "–"}:
        return None

    t = re.sub(r"\[\d+\]", "", t).replace(",", "").strip()
    t = t.replace("−", "-")

    if t.endswith("%"):
        t = t[:-1].strip()

    try:
        return float(t)
    except ValueError:
        return None


# ----------------- MediaWiki API fetch -----------------
def fetch_rendered_html_via_api(page):
    """Fetch rendered HTML for a wiki page via MediaWiki API.

    Uses action=parse with prop=text to get the page content as rendered HTML.
    Tries both /api.php and /w/api.php, which vary by MediaWiki configuration.

    Args:
        page: Page title, e.g. "Equipment_Stats".

    Returns:
        Rendered HTML string for the page content.

    Raises:
        RuntimeError: If both API endpoints fail.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-GB,en;q=0.9",
        "Referer": f"{BASE}/wiki/{page}",
    }

    last_err = None
    for api in API_CANDIDATES:
        try:
            params = {
                "action": "parse",
                "page": page,
                "prop": "text",
                "format": "json",
                "formatversion": "2",
                "redirects": "1",
            }
            r = requests.get(api, params=params, headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()

            if "error" in data:
                raise RuntimeError(f"{api} returned API error: {data['error']}")

            return data["parse"]["text"]

        except Exception as e:
            last_err = e

    raise RuntimeError(
        f"Failed to fetch rendered HTML via API endpoints. Last error: {last_err}"
    )


# ----------------- HTML parsing -----------------
def extract_heading_table_groups(rendered_html):
    """Group HTML tables by the nearest preceding heading (h2/h3).

    For each h2/h3, collects consecutive sibling <table> elements until the
    next heading. If no headings are found, all tables are returned in one group.

    Args:
        rendered_html: Rendered HTML string (from MediaWiki action=parse).

    Returns:
        List of tuples: (heading_text, [table_html, ...]).
    """
    soup = BeautifulSoup(rendered_html, "lxml")
    headings = soup.select("h2, h3")

    groups = []
    for h in headings:
        heading_text = h.get_text(" ", strip=True)
        if not heading_text:
            continue

        tables = []
        sib = h.next_sibling
        while sib is not None:
            if getattr(sib, "name", None) is None:
                sib = sib.next_sibling
                continue
            if sib.name in ("h2", "h3"):
                break
            if sib.name == "table":
                tables.append(str(sib))
            sib = sib.next_sibling

        if tables:
            groups.append((heading_text, tables))

    if not groups:
        all_tables = [str(t) for t in soup.select("table")]
        if all_tables:
            groups = [("Equipment", all_tables)]

    return groups


def table_html_to_df(table_html):
    """Convert a single HTML table string into a pandas DataFrame.

    Args:
        table_html: A string containing one <table>...</table> block.

    Returns:
        DataFrame of the first table parsed by pandas.read_html.
    """
    df = pd.read_html(table_html)[0]
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
    df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]
    return df


def df_to_parts(df):
    """Convert a DataFrame into a list of {"name": ..., "stats": {...}}.

    Assumes:
        - First column is the equipment name.
        - Remaining columns are stats.

    Args:
        df: Table DataFrame.

    Returns:
        List of dicts with keys: "name" (str) and "stats" (dict).
    """
    if df.shape[1] < 1:
        return []

    name_col = df.columns[0]
    stat_cols = list(df.columns[1:])

    out = []
    for _, row in df.iterrows():
        name = str(row.get(name_col, "")).strip()
        if not name or name.lower() == "nan":
            continue

        stats = {}
        for c in stat_cols:
            key = norm_key(c)
            if not key:
                continue
            val = to_number(row.get(c))
            if val is None:
                continue
            stats[key] = val

        out.append({"name": name, "stats": stats})

    return out


def scrape_parts_database():
    """Scrape equipment tables and return a raw database keyed by heading.

    Returns:
        Dict mapping heading/category name to list of equipment items.
        On this page, this typically returns {"EQUIPMENT": [...]} or similar,
        before postprocessing.
    """
    rendered_html = fetch_rendered_html_via_api(PAGE)
    groups = extract_heading_table_groups(rendered_html)

    db = {}
    for heading, tables in groups:
        # The page headings don't map cleanly, but we keep the old behavior:
        # collect items under the heading name transformed to an uppercase key.
        key = re.sub(r"[^A-Za-z0-9]+", "_", heading.strip()).upper()
        items = []

        for t_html in tables:
            try:
                df = table_html_to_df(t_html)
                items.extend(df_to_parts(df))
            except Exception:
                continue

        if items:
            db.setdefault(key, [])
            db[key].extend(items)

    return db


# ----------------- postprocess: names, categories, renames, ordering -----------------
def clean_item_name(name):
    """Normalize known typos/casing in item names.

    Args:
        name: Raw item name.

    Returns:
        Cleaned item name string.
    """
    fixes = {
        "Cyber Exhasut": "Cyber Exhaust",
        "Spooky engine": "Spooky Engine",
    }
    name = fixes.get(name, str(name)).strip()
    name = re.sub(r"\s+", " ", name)
    return name


def infer_category(name):
    """Infer part category from item name.

    Args:
        name: Cleaned item name.

    Returns:
        One of: ENGINE, EXHAUST, SUSPENSION, GEARBOX, TRINKET.
    """
    for suffix, cat in CATEGORY_SUFFIXES:
        if name.endswith(suffix):
            return cat

    lower = name.lower()
    if "engine" in lower:
        return "ENGINE"
    if "exhaust" in lower:
        return "EXHAUST"
    if "suspension" in lower:
        return "SUSPENSION"
    if "gearbox" in lower:
        return "GEARBOX"
    return "TRINKET"


def rename_and_filter_stats(stats):
    """Apply STAT_RENAMES and DROP_STATS to a stats dict.

    Args:
        stats: Dict of raw stats parsed from table columns.

    Returns:
        Cleaned stats dict.
    """
    out = {}
    for k, v in (stats or {}).items():
        if k in DROP_STATS:
            continue
        nk = STAT_RENAMES.get(k, k)
        out[nk] = v
    return out


def postprocess_parts_database(db):
    """Convert raw db into categorized PARTS_DATABASE with renamed stats.

    Expects the raw scrape result to contain a single key with all items
    (e.g., "EQUIPMENT" or "EQUIPMENT_STATS"). This function will merge all
    categories in the raw db and then split into ENGINE/EXHAUST/etc using
    the item name.

    Args:
        db: Raw scrape result.

    Returns:
        Categorized dict: {"ENGINE": [...], "EXHAUST": [...], ...}
    """
    # Merge all raw groups into one list (robust to headings)
    items = []
    for _, lst in (db or {}).items():
        items.extend(lst)

    out = {k: [] for k in DB_ORDER}

    for item in items:
        name = clean_item_name(item.get("name", ""))
        cat = infer_category(name)
        stats = rename_and_filter_stats(item.get("stats", {}))
        out[cat].append({"name": name, "stats": stats})

    # drop empties
    return {k: v for k, v in out.items() if v}


def reorder_db(db):
    """Reorder top-level categories to match DB_ORDER.

    Args:
        db: Categorized parts database.

    Returns:
        New dict with keys ordered like DB_ORDER.
    """
    return {k: db[k] for k in DB_ORDER if k in db}


def reorder_stats(stats):
    """Reorder stats keys to match STAT_ORDER and append unknown keys.

    Args:
        stats: Stats dict for a single item.

    Returns:
        Ordered stats dict.
    """
    if not stats:
        return {}

    present = set(stats.keys())
    ordered_keys = [k for k in STAT_ORDER if k in present]
    remaining = sorted(present - set(ordered_keys))

    out = {}
    for k in ordered_keys + remaining:
        out[k] = stats[k]
    return out


def apply_ordering(db):
    """Apply both DB ordering and per-item stat ordering.

    Args:
        db: Categorized parts database.

    Returns:
        Ordered parts database (new dict; items mutated in-place for stats).
    """
    db = reorder_db(db)
    for _, items in db.items():
        for item in items:
            item["stats"] = reorder_stats(item.get("stats", {}))
    return db


# ----------------- pretty Python writer -----------------
def fmt_number(x):
    """Format numbers to match your sample output style.

    Rules:
        - 1.0 -> 1
        - 0.2 stays 0.2
        - avoid floating noise via rounding

    Args:
        x: A numeric value.

    Returns:
        String representation.
    """
    if isinstance(x, bool):
        return "True" if x else "False"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if math.isfinite(x) and x.is_integer():
            return str(int(x))
        rounded = round(x, 10)
        s = repr(rounded)
        # trim trailing zeros in decimal
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s
    return repr(x)


def fmt_value(v):
    """Format a value for Python-source output.

    Args:
        v: Value to format.

    Returns:
        Python literal string (with double-quoted strings).
    """
    if isinstance(v, dict):
        return fmt_stats_dict(v)
    if isinstance(v, list):
        # Only used if you later store list values; not expected for stats here.
        return "[" + ", ".join(fmt_value(x) for x in v) + "]"
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, (int, float, bool)):
        return fmt_number(v)
    if v is None:
        return "None"
    return repr(v)


def fmt_stats_dict(d):
    """Format a stats dict in one line: {"T1": 0.8, "Speed": 1}.

    Args:
        d: Stats dictionary.

    Returns:
        One-line Python dict literal.
    """
    if not d:
        return "{}"
    parts = [f'"{k}": {fmt_value(v)}' for k, v in d.items()]
    return "{" + ", ".join(parts) + "}"


def fmt_item(item):
    """Format one item line: {"name": "...", "stats": {...}}.

    Args:
        item: Dict with keys "name" and "stats".

    Returns:
        Python literal string for the item dict.
    """
    name = item.get("name", "")
    stats = item.get("stats", {})
    return f'{{"name": "{name}", "stats": {fmt_stats_dict(stats)}}}'


def format_parts_database(db):
    """Create the pretty Python source for PARTS_DATABASE.

    Args:
        db: Categorized, ordered parts database.

    Returns:
        A string containing Python source code.
    """
    lines = []
    lines.append("PARTS_DATABASE = {")
    cat_indent = " " * 4
    item_indent = " " * 8

    for cat, items in db.items():
        lines.append(f'{cat_indent}"{cat}": [')
        for item in items:
            lines.append(f"{item_indent}{fmt_item(item)},")
        lines.append(f"{cat_indent}],")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def write_parts_database_py(db, out_path):
    """Write the pretty Python module containing PARTS_DATABASE.

    Args:
        db: Categorized, ordered parts database.
        out_path: Output path for the .py file.
    """
    out_path = Path(out_path)
    out_path.write_text(format_parts_database(db), encoding="utf-8")


def write_outputs(db, out_dir="scraped_out"):
    """Write JSON and pretty Python outputs.

    Args:
        db: Categorized, ordered parts database.
        out_dir: Directory to write outputs into.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "parts_database.json").write_text(
        json.dumps(db, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_parts_database_py(db, out / "parts_database.py")


if __name__ == "__main__":
    raw = scrape_parts_database()
    db = postprocess_parts_database(raw)
    db = apply_ordering(db)

    print("Categories scraped:", list(db.keys()))
    for k, v in db.items():
        print(f"{k}: {len(v)} items")

    write_outputs(db)
    print("Wrote scraped_out/parts_database.json and scraped_out/parts_database.py")
