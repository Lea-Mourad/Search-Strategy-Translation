import re
import json
import nltk

nltk.download("punkt", quiet=True)

# -------------------- Load DB rules --------------------
with open("databaserules.json", "r") as f:
    DB_RULES = json.load(f)

# -------------------- Extract all field tags and MeSH dynamically --------------------
ALL_FIELDS = {}
MESH_FIELDS = []

for db_name, db in DB_RULES.items():
    fields = db.get("fields", {})
    for k in fields.keys():
        ALL_FIELDS[k.lower()] = k  # lowercase for easy matching
    mesh = db.get("mesh", {})
    suffix = mesh.get("suffix")
    if suffix:
        MESH_FIELDS.append(suffix.lower())
    # optional: detect MeSH from JSON if needed

# -------------------- Query validation --------------------
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

# -------------------- Query normalization --------------------
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

# -------------------- Tokenization --------------------
def tokenize_query(query):
    # Match parentheses, field tags, quoted phrases, or words
    pattern = r'\(|\)|\[[^\]]*\]|"[^"]*"|\w+'
    return re.findall(pattern, query)

def merge_terms(tokens):
    merged = []
    buffer = []
    for t in tokens:
        if t.upper() in ["AND", "OR", "NOT"] or t in ["(", ")"] or t.startswith("[") or t.startswith('"'):
            if buffer:
                merged.append(" ".join(buffer))
                buffer = []
            merged.append(t)
        else:
            buffer.append(t)
    if buffer:
        merged.append(" ".join(buffer))
    return merged

# -------------------- Dynamic Token Classification --------------------
def classify_tokens(tokens):
    stream = []
    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token.upper() in ["AND", "OR", "NOT"]:
            stream.append({"type": "BOOLEAN", "value": token})
        elif token == "(":
            stream.append({"type": "LPAREN", "value": token})
        elif token == ")":
            stream.append({"type": "RPAREN", "value": token})
        else:
            field = None
            source = None

            # Check if next token is a field or MeSH
            if i + 1 < len(tokens):
                next_token_lc = tokens[i + 1].lower()
                if next_token_lc in ALL_FIELDS:
                    field = ALL_FIELDS[next_token_lc]
                    i += 1
                elif next_token_lc.startswith("[mesh"):
                    source = "MeSH"
                    i += 1

            # Remove quotes from token
            phrase = token.strip('"')
            stream.append({"type": "PHRASE", "value": phrase, "field": field, "source": source})

        i += 1
    return stream

def process_query(query):
    if not validate_parentheses(query):
        raise ValueError("Unbalanced parentheses")
    query = normalize_query(query)
    tokens = tokenize_query(query)
    tokens = merge_terms(tokens)
    return classify_tokens(tokens)

# -------------------- Generic Formatter --------------------
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

        # MeSH handling
        if source == "MeSH":
            mesh_rules = rules.get("mesh", {})
            prefix = mesh_rules.get("explode_prefix", "")
            suffix = mesh_rules.get("explode_suffix", mesh_rules.get("suffix", ""))
            quote_mesh = mesh_rules.get("quote_mesh", False)
            if quote_mesh:
                # Only quote the term, leave suffix outside
                return f"{quote}{prefix}{v}{quote}{suffix}"
            else:
                return f"{prefix}{v}{suffix}"

        # Free text with optional field tag
        text = f"{quote}{v}{quote}"
        if field:
            text += rules["fields"].get(field, "")
        return text

    return v

# -------------------- Query Reconstruction --------------------
def reconstruct_query(tokens, db):
    parts = []
    for token in tokens:
        part = format_token(token, db)
        # Attach directly to previous token if starts with : or .
        if part.startswith((':', '.')) and parts:
            parts[-1] = f"{parts[-1]}{part}"
        else:
            parts.append(part)
    return " ".join(p for p in parts if p)

# -------------------- Main --------------------
if __name__ == "__main__":
    try:
        user_query = input("Enter PubMed query: ")
        tokens = process_query(user_query)

        print("\nTranslated Queries:\n")
        for db in DB_RULES.keys():
            print(f"{db.upper()}: {reconstruct_query(tokens, db)}")

    except Exception as e:
        print("Error:", e)
