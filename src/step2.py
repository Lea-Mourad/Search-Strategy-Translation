import re
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)

# -------------------- Step 1: Parentheses validation --------------------
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
# -------------------------------------------------------------------------

# -------------------- Step 1: Normalize query --------------------------
def normalize_query(query):
    query = query.strip()
    query = re.sub(r"\s+", " ", query)
    query = re.sub(r'\b(and|or|not)\b', lambda x: x.group().upper(), query, flags=re.IGNORECASE)
    return query
# -------------------------------------------------------------------------

# -------------------- Step 1: Tokenization -----------------------------
def tokenize_query(query):
    token_pattern = r'\(|\)|\[[^\]]+\]|"[^"]*"|\w+'
    return re.findall(token_pattern, query)
# -------------------------------------------------------------------------

# -------------------- Step 1: Merge consecutive terms ------------------
def merge_terms(tokens):
    merged_tokens = []
    buffer = []

    for token in tokens:
        if token in ['AND', 'OR', 'NOT', '(', ')'] or token.startswith('['):
            if buffer:
                merged_tokens.append(' '.join(buffer))
                buffer = []
            merged_tokens.append(token)
        else:
            buffer.append(token)

    if buffer:
        merged_tokens.append(' '.join(buffer))
    return merged_tokens
# -------------------------------------------------------------------------

# -------------------- Step 1: Token classification --------------------
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
        elif re.match(r'\[[^\]]+\]', token):
            token_type = 'FIELD_TAG'
            value = token
        else:
            token_type = 'PHRASE'
            value = token.strip('"')
        token_stream.append({"token": token, "type": token_type, "value": value})
    return token_stream
# -------------------------------------------------------------------------

# -------------------- Step 1: Full process -----------------------------
def process_query(query):
    if not validate_parentheses(query):
        print("Error: Parentheses are not balanced!")
        return []
    query = normalize_query(query)
    tokens = tokenize_query(query)
    tokens = merge_terms(tokens)
    return classify_tokens(tokens)
# -------------------------------------------------------------------------

# -------------------- Step 2: Mapping to Cochrane ----------------------
mapping_dict = {
    "BOOLEAN": {"AND": "AND", "OR": "OR", "NOT": "NOT"},
    "LPAREN": {"(": "("},
    "RPAREN": {")": ")"},
    "FIELD_TAG": {"[TIAB]": ":ti", "[MeSH Terms]": ":mh"}
}

def map_tokens_only(token_stream):
    mapped_stream = []
    for token_obj in token_stream:
        token_type = token_obj["type"]
        value = token_obj["value"]
        mapped_value = mapping_dict.get(token_type, {}).get(value, value)
        mapped_stream.append({"original": value, "type": token_type, "mapped": mapped_value})
    return mapped_stream
# -------------------------------------------------------------------------

# -------------------- Step 0: Run multiple test cases ------------------
if __name__ == "__main__":
    test_queries = [
        '(myocardial infarction[MeSH Terms] OR heart attack[TIAB]) AND hypertension',
        '((diabetes[MeSH Terms] OR "type 2 diabetes"[TIAB]) AND obesity[MeSH Terms]) NOT smoking[TIAB]',
        'asthma[MeSH Terms] OR COPD[TIAB] OR bronchitis[TIAB]',
        '"heart failure"[TIAB] AND "left ventricular dysfunction"[MeSH Terms]',
        '(stroke[MeSH Terms] or "cerebrovascular accident"[TIAB]) AND (aspirin[TIAB] OR clopidogrel[TIAB])'
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test Case {i} ---")
        structured_tokens = process_query(query)
        mapped_stream = map_tokens_only(structured_tokens)

        for t in mapped_stream:
            print(f"{t['original']:35} | {t['type']:10} | {t['mapped']}")
