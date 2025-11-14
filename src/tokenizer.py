import re
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)

# -------------------- Parentheses validation --------------------
def validate_parentheses(query):
    """Check if parentheses are balanced in the query."""
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
    """Normalize spaces and capitalize Boolean operators."""
    query = query.strip()
    query = re.sub(r"\s+", " ", query)
    query = re.sub(r'\b(and|or|not)\b', lambda x: x.group().upper(), query, flags=re.IGNORECASE)
    return query
# ----------------------------------------------------------------

# -------------------- Tokenization -----------------------------
def tokenize_query(query):
    """
    Tokenize into parentheses, field tags, quoted phrases, and words.
    """
    token_pattern = r'\(|\)|\[[^\]]+\]|"[^"]*"|\w+'
    tokens = re.findall(token_pattern, query)
    return tokens
# ----------------------------------------------------------------

# -------------------- Merge consecutive TERMS into PHRASE -------
def merge_terms(tokens):
    """
    Merge consecutive TERM tokens (words) into a single PHRASE.
    Keeps parentheses, boolean operators, and field tags separate.
    """
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
# ----------------------------------------------------------------

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
        elif re.match(r'\[.*\]', token):
            token_type = 'FIELD_TAG'
            value = token
        else:
            # PHRASE (merge quoted and unquoted terms)
            token_type = 'PHRASE'
            token = token.strip('"')  # Remove quotes
            value = token

        token_stream.append({'token': token, 'type': token_type, 'value': value})
    return token_stream
# ----------------------------------------------------------------

def process_query(query):
    if not validate_parentheses(query):
        print("Error: Parentheses are not balanced!")
        return []

    query = normalize_query(query)
    tokens = tokenize_query(query)
    tokens = merge_terms(tokens)
    token_stream = classify_tokens(tokens)
    return token_stream

# -------------------- Main loop -------------------------------
if __name__ == "__main__":
    test_query = '("heart attack" OR myocardial infarction) AND (diabetes mellitus OR hypertension)'

    structured_tokens = process_query(test_query)
    print("Structured Token Stream:")
    for t in structured_tokens:
        print(f"{t['token']:25} | {t['type']:10} | {t['value']}")
