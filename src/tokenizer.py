import re
import nltk


nltk.download('punkt', quiet=True)

# -------------------- Parentheses validation --------------------
def validate_parentheses(query):
    stack = []
    for char in query:
        if char == "(":
            stack.append(char)
        elif char == ")":
            if not stack:
                return False
            stack.pop()
    return len(stack) == 0
# ----------------------------------------------------------------

# -------------------- Normalize query --------------------------
def normalize_query(query):
    query = query.strip()
    query = re.sub(r"\s+", " ", query)
    query = re.sub(r'\b(and|or|not)\b', lambda x: x.group().upper(), query, flags=re.IGNORECASE)
    return query


# -------------------- Tokenization -----------------------------
def tokenize_query(query):
    token_pattern = r'\(|\)|\[[^\]]*\]|"[^"]*"|\w+'
    return re.findall(token_pattern, query)
# ----------------------------------------------------------------

# -------------------- Merge consecutive TERMS into PHRASE -------
def merge_terms(tokens):
    merged_tokens = []
    buffer = []

    for token in tokens:
        if token in ['AND', 'OR', 'NOT', '(', ')'] or token.startswith('[') or token.startswith('"'):
            if buffer:
                merged_tokens.append(' '.join(buffer))
                buffer = []
            merged_tokens.append(token)
        else:
            buffer.append(token)

    if buffer:
        merged_tokens.append(' '.join(buffer))
    return merged_tokens


# -------------------- Token classification ---------------------
def classify_tokens(tokens):
    token_stream = []
    for token in tokens:
        if token in ['AND', 'OR', 'NOT']:
            token_type = 'BOOLEAN'
            value = token
        elif token == '(':
            token_type = 'LPAREN'
            value = token
        elif token == ')':
            token_type = 'RPAREN'
            value = token
        elif re.match(r'\[[^\]]*\]', token):
            token_type = 'FIELD_TAG'
            value = token
        elif token.startswith('"') and token.endswith('"'):
            token_type = 'PHRASE'
            value = token.strip('"')
        else:
            token_type = 'PHRASE'
            value = token
        token_stream.append({"token": token, "type": token_type, "value": value})
    return token_stream
# ----------------------------------------------------------------

def process_query(query):
    if not validate_parentheses(query):
        print("Error: Parentheses are not balanced!")
        return []

    query = normalize_query(query)
    tokens = tokenize_query(query)
    tokens = merge_terms(tokens)
    return classify_tokens(tokens)


# Database-specific field mapping
FIELD_MAPPING = {
    'cochrane': {
        '[TIAB]': ':ti,ab',
        '[MeSH Terms]': 'MeSH:',
    },
    'medline': {
        '[TIAB]': ':ti,ab',
        '[MeSH Terms]': 'MeSH:',
    },
    'embase': {
        '[TIAB]': ':ti,ab',
        '[MeSH Terms]': '/exp',
    }
}

def map_token_to_db(token, db):
    ttype = token['type']
    value = token['value']

    if ttype == 'PHRASE':
        if db == 'embase':
            return f"'{value}'"
        else:
            return f'"{value}"'
    elif ttype == 'FIELD_TAG':
        return FIELD_MAPPING[db].get(value, value)
    elif ttype == 'BOOLEAN':
        return value
    elif ttype in ['LPAREN', 'RPAREN']:
        return value
    else:
        if db == 'embase':
            return f"'{value}'"
        else:
            return f'"{value}"'

def reconstruct_query(tokens, db):
    mapped_tokens = [map_token_to_db(t, db) for t in tokens]
    return ' '.join(mapped_tokens)

if __name__ == "__main__":
    user_query = input("Enter PubMed query: ")

    structured_tokens = process_query(user_query)

    outputs = {}
    for db in ['cochrane', 'medline', 'embase']:
        outputs[db] = reconstruct_query(structured_tokens, db)

    print("\nTranslated Queries:")
    for db, q in outputs.items():
        print(f"{db.upper()}: {q}")
