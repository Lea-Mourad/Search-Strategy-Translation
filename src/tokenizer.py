import os
import re
import json
import nltk
nltk.download('punkt', quiet=True)

# Silence HF tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------- Optional LLM Setup ----------------------
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Lazy LLM (can be skipped for testing)
llm_pipeline = None
def get_llm():
    global llm_pipeline
    if llm_pipeline is None:
        model_name = "facebook/opt-125m"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=128)
    return llm_pipeline

# -------------------- MeSH translation -----------------------
mesh_cache = {}

def translate_mesh_term(term, use_llm=False):
    """Translate MeSH term to Cochrane/Embase/Medline. 
    If use_llm=False, returns simple static mapping for testing."""
    if term in mesh_cache:
        return mesh_cache[term]

    if not use_llm:
        # Updated to match official database formats
        translation = {
            "Cochrane": term,
            "Embase": f"'{term}'/exp",  # Embase uses single quotes and /exp for MeSH
            "Medline": term
        }
    else:
        prompt = f"""
Translate this MeSH term into equivalent controlled vocabulary terms.

Return JSON only with keys: "Cochrane", "Embase", "Medline"
Input: {term}
Output:
"""
        llm = get_llm()
        result = llm(prompt, max_length=200, do_sample=False)
        generated = result[0]["generated_text"].split("Output:")[-1].strip()
        try:
            translation = json.loads(generated)
        except json.JSONDecodeError:
            translation = {
                "Cochrane": term,
                "Embase": f"'{term}'/exp",
                "Medline": f"MeSH:{term}"
            }

    mesh_cache[term] = translation
    return translation

# -------------------- Query normalization -------------------
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

def normalize_query(query):
    query = query.strip()
    query = re.sub(r"\s+", " ", query)
    query = re.sub(r'\b(and|or|not)\b', lambda m: m.group().upper(), query, flags=re.IGNORECASE)
    # Fix spacing before brackets and parentheses
    query = re.sub(r'\(\s+', '(', query)
    query = re.sub(r'\s+\)', ')', query)
    query = re.sub(r'\s+(\[[^\]]+\])', r'\1', query)
    return query

# -------------------- Tokenization -------------------------
def tokenize_query(query):
    pattern = r'\(|\)|\[[^\]]*\]|"[^"]*"|\w+'
    return re.findall(pattern, query)

def merge_terms(tokens):
    merged = []
    buffer = []
    for t in tokens:
        if t in ['AND', 'OR', 'NOT', '(', ')'] or t.startswith('[') or t.startswith('"'):
            if buffer:
                merged.append(' '.join(buffer))
                buffer = []
            merged.append(t)
        else:
            buffer.append(t)
    if buffer:
        merged.append(' '.join(buffer))
    return merged

# -------------------- Token classification -----------------
def classify_tokens(tokens):
    stream = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        if token in ['AND', 'OR', 'NOT']:
            stream.append({"type": "BOOLEAN", "value": token})
        elif token == '(':
            stream.append({"type": "LPAREN", "value": token})
        elif token == ')':
            stream.append({"type": "RPAREN", "value": token})
        elif token.startswith('"') and token.endswith('"'):
            # Check if next token is a field tag
            if i + 1 < len(tokens) and tokens[i+1].startswith('['):
                if tokens[i+1] == '[MeSH Terms]':
                    stream.append({
                        "type": "PHRASE", 
                        "value": token.strip('"'),
                        "source": "MeSH"
                    })
                else:
                    stream.append({"type": "PHRASE", "value": token.strip('"')})
                    stream.append({"type": "FIELD_TAG", "value": tokens[i+1]})
                i += 1  # Skip the field tag
            else:
                stream.append({"type": "PHRASE", "value": token.strip('"')})
        elif token.startswith('['):
            stream.append({"type": "FIELD_TAG", "value": token})
        else:
            # Regular term
            if i + 1 < len(tokens) and tokens[i+1].startswith('['):
                if tokens[i+1] == '[MeSH Terms]':
                    stream.append({
                        "type": "PHRASE", 
                        "value": token,
                        "source": "MeSH"
                    })
                else:
                    stream.append({"type": "PHRASE", "value": token})
                    stream.append({"type": "FIELD_TAG", "value": tokens[i+1]})
                i += 1  # Skip the field tag
            else:
                stream.append({"type": "PHRASE", "value": token})
        i += 1
    return stream

def process_query(query):
    if not validate_parentheses(query):
        raise ValueError("Unbalanced parentheses")
    query = normalize_query(query)
    tokens = tokenize_query(query)
    tokens = merge_terms(tokens)
    return classify_tokens(tokens)

# -------------------- Database-specific formatting ------------------------
def format_for_cochrane(token, use_llm=False):
    """Format token for Cochrane database"""
    t = token['type']
    v = token['value']
    
    if t == 'PHRASE' and token.get('source') == 'MeSH':
        translation = translate_mesh_term(v, use_llm)
        return translation.get('Cochrane', v)
    elif t == 'PHRASE':
        return f'"{v}"'
    elif t == 'FIELD_TAG':
        if v == '[TI]':
            return ':ti'
        elif v == '[AB]':
            return ':ab'
        elif v == '[TIAB]':
            return ':ti,ab'
        elif v == '[MeSH Terms]':
            return 'MeSH:'
    elif t in ['BOOLEAN', 'LPAREN', 'RPAREN']:
        return v
    return v

def format_for_embase(token, use_llm=False):
    """Format token for Embase database - single quotes and no space before colon"""
    t = token['type']
    v = token['value']
    
    if t == 'PHRASE' and token.get('source') == 'MeSH':
        translation = translate_mesh_term(v, use_llm)
        return translation.get('Embase', f"'{v}'/exp")
    elif t == 'PHRASE':
        # Embase uses single quotes for phrases
        return f"'{v}'"
    elif t == 'FIELD_TAG':
        # Remove space before colon
        if v == '[TI]':
            return ':ti'
        elif v == '[AB]':
            return ':ab'
        elif v == '[TIAB]':
            return ':ti,ab,kw'  # add :kw for keywords
        elif v == '[MeSH Terms]':
            return '/exp'
    elif t in ['BOOLEAN', 'LPAREN', 'RPAREN']:
        return v
    return v



def format_for_medline(token, use_llm=False):
    """Format token for MEDLINE (Ovid)"""
    
    t = token['type']
    v = token['value']

    # MeSH terms
    if t == 'PHRASE' and token.get('source') == 'MeSH':
        translation = translate_mesh_term(v, use_llm)

        # Prefer exploded MeSH if available
        medline_term = translation.get('Medline', v)
        return f"exp {medline_term}/"

    # Free-text phrases
    elif t == 'PHRASE':
        return f'"{v}"'

    # Field tags → Ovid syntax
    elif t == 'FIELD_TAG':
        if v == '[TI]':
            return '.ti.'
        elif v == '[AB]':
            return '.ab.'
        elif v == '[TIAB]':
            return '.ti,ab.'
        elif v == '[MeSH Terms]':
            # handled via PHRASE + source == MeSH
            return ''

    # Boolean logic & parentheses
    elif t in ['BOOLEAN', 'LPAREN', 'RPAREN']:
        return v

    return v


# -------------------- Query reconstruction -----------------
def reconstruct_query(tokens, db, use_llm=False):
    if db == 'cochrane':
        formatter = format_for_cochrane
    elif db == 'embase':
        formatter = format_for_embase
    elif db == 'medline':
        formatter = format_for_medline
    else:
        formatter = format_for_cochrane

    parts = []

    for token in tokens:
        part = formatter(token, use_llm)

        # Attach field tags directly to previous phrase
        if part.startswith((':', '.')) and parts:
            parts[-1] = f"{parts[-1]}{part}"
        else:
            parts.append(part)

    return ' '.join(p for p in parts if p)



# -------------------- Main ---------------------------------
if __name__ == "__main__":
    try:
        user_query = input("Enter PubMed query: ")
        tokens = process_query(user_query)
        print("\nTranslated Queries:")
        
        print(f"COCHRANE: {reconstruct_query(tokens, 'cochrane', use_llm=False)}")
        print(f"EMBASE: {reconstruct_query(tokens, 'embase', use_llm=False)}")
        print(f"MEDLINE: {reconstruct_query(tokens, 'medline', use_llm=False)}")
        
    except Exception as e:
        print(f"Error: {e}")