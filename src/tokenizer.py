import re
import json
import nltk

nltk.download("punkt", quiet=True)

# ==================== Load DB rules ====================
with open("databaserules.json", "r") as f:
    DB_RULES = json.load(f)

# ==================== Field aliases ====================
FIELD_ALIASES = {
    # Title / Abstract
    "[title]": "[TI]",
    "[ti]": "[TI]",
    "[abstract]": "[AB]",
    "[ab]": "[AB]",
    "[title/abstract]": "[TIAB]",
    "[tiab]": "[TIAB]",

    # Text word
    "[text word]": "[TW]",
    "[tw]": "[TW]",

    # MeSH
    "[mesh]": "[MH]",
    "[mesh term]": "[MH]",
    "[mesh terms]": "[MH]",
    "[mh]": "[MH]",

    "[mesh major topic]": "[MAJR]",
    "[major topic]": "[MAJR]",
    "[majr]": "[MAJR]",

    "[mesh subheading]": "[SH]",
    "[subheading]": "[SH]",
    "[sh]": "[SH]",

    "[supplementary concept]": "[NM]",
    "[nm]": "[NM]",

    "[pharmacological action]": "[PA]",
    "[pa]": "[PA]",

    # Other fields
    "[other term]": "[OT]",
    "[ot]": "[OT]",
    "[pagination]": "[PG]",
    "[pg]": "[PG]",
    "[publication type]": "[PT]",
    "[pt]": "[PT]",
    "[publisher]": "[PB]",
    "[pb]": "[PB]",
    "[secondary source id]": "[SI]",
    "[si]": "[SI]",
    "[subject personal name]": "[PS]",
    "[ps]": "[PS]",
    "[transliterated title]": "[TT]",
    "[tt]": "[TT]",
    "[volume]": "[VI]",
    "[vi]": "[VI]"
}

ALL_FIELDS = {k: v for k, v in FIELD_ALIASES.items()}

# ==================== Validation ====================
def validate_parentheses(query):
    stack = []
    for c in query:
        if c == "(":
            stack.append(c)
        elif c == ")":
            if not stack:
                return False
            stack.pop()
    return not stack

# ==================== Normalization ====================
def normalize_query(query):
    query = query.strip()
    query = re.sub(r"\s+", " ", query)
    query = re.sub(
        r"\b(and|or|not)\b",
        lambda m: m.group().upper(),
        query,
        flags=re.IGNORECASE
    )
    query = re.sub(r"\(\s+", "(", query)
    query = re.sub(r"\s+\)", ")", query)
    return query

# ==================== Tokenization ====================
def tokenize_query(query):
    pattern = r'\(|\)|"[^"]*"|\[[^\]]*\]|\w+'
    return re.findall(pattern, query)

# ==================== Classification ====================
def classify_tokens(tokens):
    stream = []
    i = 0

    while i < len(tokens):
        token = tokens[i]

        # Boolean operators
        if token.upper() in ["AND", "OR", "NOT"]:
            stream.append({"type": "BOOLEAN", "value": token})
            i += 1
            continue

        # Parentheses
        if token == "(":
            stream.append({"type": "LPAREN", "value": token})
            i += 1
            continue

        if token == ")":
            stream.append({"type": "RPAREN", "value": token})
            i += 1
            continue

        # Term / Phrase
        phrase = token.strip('"')
        field = None
        source = None

        # Lookahead for field
        if i + 1 < len(tokens) and tokens[i + 1].startswith("["):
            tag = tokens[i + 1].lower()

            if tag in ALL_FIELDS:
                field = ALL_FIELDS[tag]
                if field in ["[MH]", "[MAJR]", "[SH]", "[NM]"]:
                    source = "MeSH"
                i += 1

        stream.append({
            "type": "PHRASE",
            "value": phrase,
            "field": field,
            "source": source
        })

        i += 1

    return stream

# ==================== Formatter ====================
def format_token(token, db):
    rules = DB_RULES[db]
    quote = rules.get("quote", '"')

    t = token["type"]
    v = token["value"]

    if t in ["BOOLEAN", "LPAREN", "RPAREN"]:
        return v

    if t == "PHRASE":
        field = token.get("field")
        source = token.get("source")
        
        # Skip unsupported fields completely
        unsupported = rules.get("unsupported_fields", [])
        if field in unsupported:
            return None

        db_fields = rules.get("fields", {})

        # ---------- MeSH handling ----------
        if source == "MeSH":
            if field not in db_fields:
                # Downgrade unsupported MeSH to free text
                return f"{quote}{v}{quote}"

            # Use DB-specific suffix for MeSH
            suffix = db_fields[field]  # e.g., /exp or /exp/mj
            prefix = rules.get("mesh", {}).get("explode_prefix", "")
            quote_mesh = rules.get("mesh", {}).get("quote_mesh", False)

            if quote_mesh:
                return f"{quote}{prefix}{v}{quote}{suffix}"
            else:
                return f"{prefix}{v}{suffix}"

        # ---------- Free text ----------
        text = f"{quote}{v}{quote}"
        if field and field in db_fields:
            text += db_fields[field]

        return text

    return v

# ==================== Reconstruction ====================
def reconstruct_query(tokens, db):
    parts = []
    token_count = len(tokens)

    for i, token in enumerate(tokens):
        part = format_token(token, db)

        if part is None:
            # Skip the token and remove booleans around it
            # Remove previous boolean if it exists
            if parts and parts[-1].upper() in ["AND", "OR", "NOT"]:
                parts.pop()
            # Skip next token if it’s a boolean
            if i + 1 < token_count:
                next_token = tokens[i + 1]
                next_part = format_token(next_token, db)
                if next_part and next_part.upper() in ["AND", "OR", "NOT"]:
                    # Skip next boolean by incrementing i in loop
                    tokens[i + 1]["skip"] = True
            continue

        if token.get("skip"):
            continue

        # Avoid repeated booleans
        if parts and parts[-1].upper() in ["AND", "OR", "NOT"] and part.upper() in ["AND", "OR", "NOT"]:
            continue

        parts.append(part)

    # Clean up dangling booleans at start or end
    while parts and parts[0].upper() in ["AND", "OR", "NOT"]:
        parts.pop(0)
    while parts and parts[-1].upper() in ["AND", "OR", "NOT"]:
        parts.pop()

    return " ".join(parts)


# ==================== Pipeline ====================
def process_query(query):
    if not validate_parentheses(query):
        raise ValueError("Unbalanced parentheses")

    query = normalize_query(query)
    tokens = tokenize_query(query)
    return classify_tokens(tokens)

# ==================== Main ====================
if __name__ == "__main__":
    try:
        user_query = input("Enter PubMed query: ")
        tokens = process_query(user_query)

        print("\nTranslated Queries:\n")
        for db in DB_RULES.keys():
            print(f"{db.upper()}: {reconstruct_query(tokens, db)}")

    except Exception as e:
        print("Error:", e)
