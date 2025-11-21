"""
Fuzzy matching helper to map user food input to closest food in food_nutrition_data_final.csv.
Enhancements:
 - Aggressive normalization pipeline (remove diacritics, handle teencode, slang, abbreviations)
 - Auto-generated aliases (initialisms, concatenated names, brand slang) for every food entry
 - Multi-candidate lookup that can return multiple plausible dishes
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Iterable

import pandas as pd
import unicodedata
from rapidfuzz import process, fuzz


DIGIT_TRANSLATION = str.maketrans({
    "0": "o",
    "1": "i",
    "2": "h",
    "3": "e",
    "4": "a",
    "5": "s",
    "6": "g",
    "7": "t",
    "8": "b",
    "9": "g",
})

REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{2,}")

TEENCODE_MAP = {
    "bm": "banh mi",
    "bmi": "banh mi",
    "banhmy": "banh mi",
    "banhmj": "banh mi",
    "bbh": "bun bo hue",
    "bb": "bun bo",
    "bunbo": "bun bo",
    "bunbohue": "bun bo hue",
    "bbq": "nuong",
    "ct": "com tam",
    "comtam": "com tam",
    "cogh": "com ga hoi an",
    "cq": "com quay",
    "cph": "com phan",
    "pk": "phao keo",
    "pho": "pho",
    "fo": "pho",
    "tra": "tra",
    "ts": "tra sua",
    "trs": "tra sua",
    "pizza": "pizza",
    "pz": "pizza",
    "burger": "burger",
    "hamberger": "burger",
    "hamber": "burger",
    "hb": "burger",
    "banhbao": "banh bao",
    "banhbeo": "banh beo",
    "xien": "xien",
    "xienque": "xien que",
    "xq": "xien que",
    "kfc": "ga ran",
    "lotteria": "ga ran",
    "mac": "macdonald",
    "mcd": "macdonald",
    "mc": "macdonald",
    "coke": "nuoc ngot",
    "coca": "nuoc ngot",
    "pepsi": "nuoc ngot",
    "sting": "nuoc ngot",
    "7up": "nuoc ngot",
    "nuoc ngot": "nuoc ngot",
    "soda": "nuoc ngot",
    "redbull": "nuoc tang luc",
    "st": "sting",
    "nau": "ca phe sua",
    "den": "ca phe den",
    "cf": "ca phe",
    "caphe": "ca phe",
    "cafe": "ca phe",
}

PHRASE_SYNONYMS = {
    "ga ran": "ga chien",
    "ga quay": "ga nuong",
    "ga nuong mat ong": "ga nuong",
    "chicken": "ga",
    "fried chicken": "ga chien",
    "pork": "thit heo",
    "pig": "thit heo",
    "heo quay": "thit heo quay",
    "beef": "thit bo",
    "steak": "thit bo",
    "salmon": "ca hoi",
    "tuna": "ca ngu",
    "shrimp": "tom",
    "squid": "muc",
    "octopus": "bach tuoc",
    "egg": "trung",
    "vegetable": "rau",
    "veggie": "rau",
    "vegan": "chay",
    "rice": "com",
    "noodle": "mi",
    "noodles": "mi",
    "instant noodle": "mi tom",
    "instant noodles": "mi tom",
    "ramen": "mi",
    "coffee": "ca phe",
    "milk tea": "tra sua",
    "bubble tea": "tra sua",
    "sushi": "sushi",
}

STOPWORDS = {
    "mon",
    "an",
    "monan",
    "mon an",
    "thuc",
    "pham",
    "do",
    "monan",
    "thit",
    "món",
    "ăn",
}

def _strip_diacritics(s: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'
    )


def _collapse_repeats(token: str) -> str:
    return REPEATED_CHAR_PATTERN.sub(r"\1", token)


def _replace_digits(text: str) -> str:
    return text.translate(DIGIT_TRANSLATION)


def _sanitize(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", text)


def _apply_phrase_synonyms(text: str) -> str:
    normalized = text
    # replace longer phrases first
    for alias, canonical in sorted(PHRASE_SYNONYMS.items(), key=lambda x: -len(x[0])):
        normalized = normalized.replace(alias, canonical)
    return normalized


def _normalize_token(token: str) -> str:
    token = _collapse_repeats(token)
    if not token:
        return ""

    replacement = TEENCODE_MAP.get(token)
    if replacement:
        return replacement

    # handle phonetic/letter substitutions
    if token.startswith("f"):
        token = "ph" + token[1:]
    if token.startswith("w"):
        token = "qu" + token[1:]
    if token.startswith("j"):
        token = "gi" + token[1:]
    if token.startswith("dz"):
        token = "d" + token[2:]
    if token.endswith("j"):
        token = token[:-1] + "i"

    return token


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = _strip_diacritics(s)
    s = _replace_digits(s)
    s = _apply_phrase_synonyms(s)
    s = _sanitize(s)
    tokens: list[str] = []
    for tok in s.split():
        normalized = _normalize_token(tok)
        if normalized:
            tokens.extend(normalized.split())
    return ' '.join(tokens)


def _generate_aliases(name: str, normalized: str) -> set[str]:
    tokens = [tok for tok in normalized.split() if tok and tok not in STOPWORDS]
    alias_set: set[str] = set()

    if normalized:
        alias_set.add(normalized)
        alias_set.add(normalized.replace(" ", ""))

    if tokens:
        alias_set.add(' '.join(tokens))
        alias_set.add(''.join(tokens))

    initials = ''.join(tok[0] for tok in tokens if tok)
    if len(initials) >= 2:
        alias_set.add(initials)

    for tok in tokens:
        if len(tok) < 2:
            continue
        alias_set.add(tok)
        if tok.startswith("ph"):
            alias_set.add("f" + tok[2:])
        if tok.startswith("qu"):
            alias_set.add("w" + tok[2:])

    return {alias for alias in alias_set if alias}


def _select_best_name(query_key: str, candidate_names: Iterable[str]) -> str | None:
    best_name = None
    best_score = -1
    for name in candidate_names:
        normalized_name = name_normalized_map.get(name)
        if not normalized_name:
            normalized_name = normalize_text(name)
            name_normalized_map[name] = normalized_name
        score = fuzz.WRatio(query_key, normalized_name)
        if score > best_score:
            best_name = name
            best_score = score
    return best_name


def _row_to_payload(row: pd.Series, score: float | None = None) -> dict:
    # Try to get image_url from either "image_url" or "Link" column (for backward compatibility)
    image_url = None
    if "image_url" in row.index:
        image_url = row["image_url"] if pd.notna(row["image_url"]) else None
    elif "Link" in row.index:
        image_url = row["Link"] if pd.notna(row["Link"]) else None
    
    return {
        "name": row["Tên thực phẩm"],
        "score": score,
        "calories": float(row["Năng lượng"]) if pd.notna(row["Năng lượng"]) else 0.0,
        "protein": float(row["Protein"]) if pd.notna(row["Protein"]) else 0.0,
        "carbs": float(row["Glucid"]) if pd.notna(row["Glucid"]) else 0.0,
        "fat": float(row["Lipid"]) if pd.notna(row["Lipid"]) else 0.0,
        "fiber": float(row["Celluloza"]) if pd.notna(row["Celluloza"]) else 0.0,
        "weight_g": 100.0,
        "image_url": image_url if image_url and str(image_url).strip() else None,
    }


# Load Vietnamese food dataset
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "food_nutrition_data_final.csv")
food_master = pd.read_csv(DATA_PATH, encoding='utf-8')

food_names = food_master["Tên thực phẩm"].fillna("").astype(str).tolist()
name_normalized_map = {name: normalize_text(name) for name in food_names}

choices_map: dict[str, list[str]] = defaultdict(list)
choices: list[str] = []
for name in food_names:
    normalized = name_normalized_map[name]
    aliases = _generate_aliases(name, normalized)
    for alias in aliases:
        if name not in choices_map[alias]:
            choices_map[alias].append(name)
        if alias not in choices:
            choices.append(alias)


def _collect_matches(
    key: str,
    nutrition_df: pd.DataFrame,
    limit: int,
    threshold: int | None,
) -> list[dict]:
    best_matches = process.extract(key, choices, scorer=fuzz.WRatio, limit=limit)
    results: list[dict] = []
    for matched_key, score, _ in best_matches:
        if threshold is not None and score < threshold:
            continue
        candidate_names = choices_map.get(matched_key, [])
        best_name = _select_best_name(key, candidate_names)
        if not best_name:
            continue
        matched_rows = nutrition_df[nutrition_df["Tên thực phẩm"] == best_name]
        if matched_rows.empty:
            continue
        results.append(_row_to_payload(matched_rows.iloc[0], score=score))
    return results


def _deduplicate_matches(matches: list[dict]) -> list[dict]:
    deduped: dict[str, dict] = {}
    for match in matches:
        name = match["name"]
        current = deduped.get(name)
        if not current or (match.get("score") or 0) > (current.get("score") or 0):
            deduped[name] = match
    return list(deduped.values())


def map_food_candidates(
    text: str,
    nutrition_df: pd.DataFrame,
    threshold: int = 55,
    top_n: int = 5,
) -> list[dict]:
    if not text:
        return []

    key = normalize_text(text)
    if not key:
        return []

    limit = max(top_n * 2, 5)

    results = _collect_matches(key, nutrition_df, limit=limit, threshold=threshold)
    if results:
        return _deduplicate_matches(results)[:top_n]

    relaxed_threshold = max(threshold - 15, 35)
    results = _collect_matches(key, nutrition_df, limit=limit, threshold=relaxed_threshold)
    if results:
        return _deduplicate_matches(results)[:top_n]

    # Ultimate fallback: allow best suggestion even when score below threshold
    results = _collect_matches(key, nutrition_df, limit=limit, threshold=None)
    if results:
        return _deduplicate_matches(results)[:top_n]

    tokens = key.split()
    for tok in tokens:
        if len(tok) < 1:
            continue
        token_results = _collect_matches(tok, nutrition_df, limit=limit, threshold=threshold)
        if token_results:
            return _deduplicate_matches(token_results)[:top_n]
        token_results = _collect_matches(tok, nutrition_df, limit=limit, threshold=None)
        if token_results:
            return _deduplicate_matches(token_results)[:top_n]

    return []


def map_food(text: str, nutrition_df: pd.DataFrame, threshold: int = 55):
    matches = map_food_candidates(text, nutrition_df, threshold=threshold, top_n=1)
    return matches[0] if matches else None


if __name__ == "__main__":
    print(map_food("banhmj thjt nuong"))
