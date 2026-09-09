"""Tokenización sin descargar corpus y representación dispersa de reportes."""
import json
import re
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

TOKENIZER = RegexpTokenizer(r'[A-Za-zÀ-ÖØ-öø-ÿ]+')
STOP = set('the and for with from this that was were are has have had its of in to a an as on at by or be is it not which during'.split())
STOP.update('grupo galicia financial report quarter million income'.split())


def tokenizar(text):
    return [t.lower() for t in TOKENIZER.tokenize(text) if len(t)>2 and t.lower() not in STOP]


def analizar(carpeta: Path, out: Path):
    from pypdf import PdfReader
    manifest = json.loads((carpeta / 'fuentes.json').read_text(encoding='utf-8'))
    documents, rows, sources = [], [], []
    for source in manifest:
        path = carpeta / source['archivo']
        if not path.is_file():
            raise FileNotFoundError(f'Falta {path}. Consulta datos/README.md para descargar la fuente.')
        if path.suffix.lower() == '.pdf':
            pages = [page.extract_text() or '' for page in PdfReader(path).pages]
        else:
            pages = [path.read_text(encoding='utf-8')]
        sources.append({**source, 'paginas': len(pages)})
        for page_number, text in enumerate(pages, 1):
            tokens = tokenizar(text)
            if len(tokens) < 10:
                continue
            documents.append(text)
            amounts = re.findall(r'Ps\.\s*[\d,]+(?:\.\d+)?\s*million', text, flags=re.I)
            percentages = re.findall(r'(?<!\w)\d+(?:\.\d+)?\s*%', text)
            rows.append({'archivo': path.name, 'pagina': page_number,
                         'tokens': len(tokens), 'terminos_distintos': len(set(tokens)),
                         'montos_detectados': '; '.join(dict.fromkeys(amounts)),
                         'porcentajes_detectados': '; '.join(dict.fromkeys(percentages))})
    if not documents:
        raise ValueError('No se encontró texto suficiente en los reportes; un PDF escaneado necesita OCR.')
    vectorizer = TfidfVectorizer(tokenizer=tokenizar, token_pattern=None, lowercase=False)
    matrix = sparse.csr_matrix(vectorizer.fit_transform(documents))
    sparse.save_npz(out / 'nlp_tfidf.npz', matrix)
    words = vectorizer.get_feature_names_out()
    means = np.asarray(matrix.mean(axis=0)).ravel()
    counts = Counter(t for doc in documents for t in tokenizar(doc))
    terms = pd.DataFrame({'termino': words, 'tfidf_medio': means,
                          'frecuencia': [counts[w] for w in words]}).sort_values('tfidf_medio', ascending=False)
    terms.to_csv(out / 'nlp_terminos.csv', index=False)
    for i, row in enumerate(rows):
        indices = matrix.getrow(i).toarray().ravel().argsort()[-5:][::-1]
        row['terminos_principales'] = ', '.join(words[indices])
    pd.DataFrame(rows).to_csv(out / 'nlp_paginas.csv', index=False)
    (out / 'nlp_vocabulario.json').write_text(json.dumps(words.tolist(), ensure_ascii=False, indent=2), encoding='utf-8')
    metadata = {'fuentes': sources, 'paginas_analizadas': len(rows), 'vocabulario': len(words),
                'tokens': sum(counts.values()), 'memoria_dispersa_bytes': int(matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes),
                'memoria_densa_equivalente_bytes': int(matrix.shape[0]*matrix.shape[1]*8),
                'alcance': 'Tokenización y extracción descriptiva; no clasifica sentimiento ni alimenta los modelos bursátiles.'}
    return metadata, terms
