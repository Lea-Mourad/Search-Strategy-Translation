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

    def is_boolean(tok):
        return tok.upper() in ["AND", "OR", "NOT"]

    def is_paren(tok):
        return tok in ["(", ")"]

    def is_field(tok):
        return tok.startswith("[") and tok.endswith("]")

    while i < len(tokens):
        tok = tokens[i]

        # Boolean / parentheses
        if is_boolean(tok):
            stream.append({"type": "BOOLEAN", "value": tok.upper()})
            i += 1
            continue
        if tok == "(":
            stream.append({"type": "LPAREN", "value": tok})
            i += 1
            continue
        if tok == ")":
            stream.append({"type": "RPAREN", "value": tok})
            i += 1
            continue

        # Term/phrase (quoted OR unquoted words possibly multiword)
        term_parts = []
        field = None
        source = None

        # If it's a quoted phrase, keep it as one unit
        if tok.startswith('"') and tok.endswith('"'):
            term_parts = [tok.strip('"')]
            i += 1
        else:
            # Collect consecutive word tokens into one term
            while i < len(tokens) and (not is_boolean(tokens[i])) and (not is_paren(tokens[i])) and (not is_field(tokens[i])) and (not (tokens[i].startswith('"') and tokens[i].endswith('"'))):
                term_parts.append(tokens[i])
                i += 1

            # If next token is a quoted phrase, treat it separately (rare edge case)
            if i < len(tokens) and tokens[i].startswith('"') and tokens[i].endswith('"'):
                # flush collected words first
                pass

        phrase = " ".join(term_parts).strip()

        # Attach field tag if it comes next
        if i < len(tokens) and is_field(tokens[i]):
            tag = tokens[i].lower()
            if tag in ALL_FIELDS:
                field = ALL_FIELDS[tag]
                if field in ["[MH]", "[MAJR]", "[SH]", "[NM]"]:
                    source = "MeSH"
                i += 1  # consume the field tag

        stream.append({"type": "PHRASE", "value": phrase, "field": field, "source": source})

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
               # ---------- MeSH handling ----------
        if source == "MeSH":

            # ✅ OVID MEDLINE FIX (ONLY)
            if db.lower() == "medline":
                # Major Topic = leading *
                if field == "[MAJR]":
                    return f"*{v.title()}/"
                else:
                    return f"{v.title()}/"

            # ---- all other databases UNCHANGED ----
            if field not in db_fields:
                # Downgrade unsupported MeSH to free text
                return f"{quote}{v}{quote}"

            suffix = db_fields[field]
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

def reconstruct_query(tokens, db):
    parts = []
    skip_indices = set()
    token_count = len(tokens)

    for i, token in enumerate(tokens):
        if i in skip_indices:
            continue

        part = format_token(token, db)

        if part is None:
            # Skip the token and remove booleans around it
            if parts and parts[-1].upper() in ["AND", "OR", "NOT"]:
                parts.pop()

            # Skip next token if it’s a boolean (without mutating tokens)
            if i + 1 < token_count:
                next_token = tokens[i + 1]
                if next_token["type"] == "BOOLEAN":
                    skip_indices.add(i + 1)
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
