import re
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)  # Needed for word_tokenize

# Example MeSH dictionary (replace with a full MeSH list later)
MESH_TERMS = {"heart attack", "myocardial infarction", "diabetes mellitus", "hypertension"}

# -------------------- FIX 1: Parentheses validation --------------------
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
# -----------------------------------------------------------------------

# -------------------- FIX 2: Normalize terms (lowercase) ----------------
def normalize_query(query):
    """Normalize spaces, Boolean operators, and lowercase terms."""
    query = query.strip()
    query = re.sub(r"\s+", " ", query)
    # Capitalize Boolean operators
    query = re.sub(r'\b(and|or|not)\b', lambda x: x.group().upper(), query, flags=re.IGNORECASE)
    return query
# -----------------------------------------------------------------------

# -------------------- FIX 3: Tokenization using regex ------------------
def tokenize_query(query):
    """
    Tokenize into parentheses, field tags, quoted phrases, and terms.
    Multi-word phrases not in quotes will be detected later.
    """
    token_pattern = r'\(|\)|\[[^\]]+\]|"[^"]*"|\w+'
    tokens = re.findall(token_pattern, query)
    return tokens
# -----------------------------------------------------------------------

# -------------------- FIX 4: Multi-word phrase detection ----------------
def detect_multi_word_phrases(tokens):
    """
    Merge consecutive TERM tokens if they are in the MeSH dictionary.
    Example: ['myocardial', 'infarction'] -> ['myocardial infarction']
    """
    merged_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        # Skip anything that is not a word (Boolean, parentheses, quotes, field tags)
        if token in ['AND', 'OR', 'NOT'] or token in ['(', ')'] or token.startswith('"') or token.startswith('['):
            merged_tokens.append(token)
            i += 1
            continue

        # Now we’re sure token is a normal word
        phrase = token
        j = i + 1
        merged = False
        while j < len(tokens):
            next_token = tokens[j]
            if re.match(r'^\w+$', next_token):
                phrase_candidate = phrase + " " + next_token
                if phrase_candidate.lower() in MESH_TERMS:
                    phrase = phrase_candidate
                    j += 1
                    merged = True
                else:
                    break
            else:
                break
        merged_tokens.append(phrase)
        i = j if merged else i + 1

    return merged_tokens
# -----------------------------------------------------------------------

# -------------------- FIX 5: Token classification with MeSH detection ----------------
def classify_tokens(tokens):
    token_stream = []
    for token in tokens:
        # Boolean operators
        if token in ['AND', 'OR', 'NOT']:
            token_type = 'BOOLEAN'
        # Parentheses
        elif token == '(':
            token_type = 'LPAREN'
        elif token == ')':
            token_type = 'RPAREN'
        # Field tags
        elif re.match(r'\[.*\]', token):
            token_type = 'FIELD_TAG'
        # Quoted phrases
        elif '"' in token:
            token_type = 'PHRASE'
        # Multi-word MeSH terms
        elif token.lower() in MESH_TERMS:
            token_type = 'MESH_TERM'
        # All other terms
        else:
            token_type = 'TERM'
        token_stream.append({'token': token, 'type': token_type})
    return token_stream
# -----------------------------------------------------------------------

def process_query(query):
    # Validate parentheses first
    if not validate_parentheses(query):
        print("Error: Parentheses are not balanced!")
        return []

    # Normalize spaces and Boolean operators
    query = normalize_query(query)

    # Tokenize
    tokens = tokenize_query(query)

    # Detect multi-word phrases from MeSH dictionary
    tokens = detect_multi_word_phrases(tokens)

    # Classify tokens
    token_stream = classify_tokens(tokens)
    return token_stream

# -------------------- Main loop --------------------
if __name__ == "__main__":
    # Example query for testing 
    test_query = '("heart attack" OR myocardial infarction) AND (diabetes mellitus OR hypertension)'


    print(f"Testing query:\n{test_query}\n")
    structured_tokens = process_query(test_query)
    if structured_tokens:
        print("Structured Token Stream:")
        for t in structured_tokens:
            print(f"{t['token']:25} | {t['type']}")
