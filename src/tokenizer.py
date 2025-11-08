import re
import nltk #this is an nlp tool kit 
from nltk.tokenize import word_tokenize #this is a tokeniser


nltk.download('punkt') #helps split words into words and punctuation correctly

def normalize_query(query):
    """Normalize spaces and Boolean operators."""
    query = query.strip()                  # remove leading/trailing spaces
    query = re.sub(r"\s+", " ", query)    # collapse multiple spaces
    # Capitalize Boolean operators
    query = re.sub(r'\b(and|or|not)\b', lambda x: x.group().upper(), query, flags=re.IGNORECASE)
    return query

def tokenize_query(query):
    """Tokenize query into terms, phrases, parentheses, and field tags."""
    token_pattern = r'\(|\)|\[[^\]]+\]|"[^"]*"|\S+'
    tokens = re.findall(token_pattern, query)
    return tokens

def classify_tokens(tokens):
    """Classify each token by type."""
    token_stream = []
    for token in tokens:
        if token in ['AND', 'OR', 'NOT']:
            token_type = 'BOOLEAN'
        elif token == '(':
            token_type = 'LPAREN'
        elif token == ')':
            token_type = 'RPAREN'
        elif re.match(r'\[.*\]', token):
            token_type = 'FIELD_TAG'
        elif '"' in token:
            token_type = 'PHRASE'
        else:
            token_type = 'TERM'
        token_stream.append({'token': token, 'type': token_type})
    return token_stream

# this takes a raw query and returns a structured token stream.
def process_query(query):
    query = normalize_query(query)
    tokens = tokenize_query(query)
    token_stream = classify_tokens(tokens)
    return token_stream


# Main function to handle dynamic input
if __name__ == "__main__":
    print("Enter your PubMed query (or type 'exit' to quit):")
    while True:
        user_input = input(">>> ")
        if user_input.lower() == "exit":
            break
        structured_tokens = process_query(user_input)
        print("Structured Token Stream:")
        for t in structured_tokens:
            print(f"{t['token']:25} | {t['type']}")
        print("\n")  
