# mesh_indexing.py

from tokenizer import process_query

def extract_mesh_terms(pubmed_query):
    """
    STEP 1:
    Extract MeSH-indexed terms from a PubMed query
    """
    tokens = process_query(pubmed_query)

    mesh_terms = []
    for token in tokens:
        if token.get("source") == "MeSH":
            mesh_terms.append(token["value"])

    # remove duplicates, preserve order
    return list(dict.fromkeys(mesh_terms))


if __name__ == "__main__":
    query = input("Enter PubMed query: ")
    mesh_terms = extract_mesh_terms(query)

    print("\nSTEP 1 — Indexed (MeSH) terms found:")
    for term in mesh_terms:
        print(f"- {term}")
