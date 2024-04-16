import faiss
import hashlib
import json
import logging
import numpy as np
import os
import re
import requests

from bs4 import BeautifulSoup
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from typing import List
from urllib.parse import urlparse

VECTOR_DIMENSION = 384
INDEX_FILE = "vector.index"
METADATA_FILE = "metadata.json"
SENTENCE_CHUNK_SIZE = 5


class VectorManager:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = self._initialize_index()
        self.metadata, self.hash_set = self._load_metadata()

    def _initialize_index(self):
        index = faiss.IndexFlatL2(VECTOR_DIMENSION)
        if os.path.exists(INDEX_FILE):
            index = faiss.read_index(INDEX_FILE)
        return index

    def _load_metadata(self):
        if not os.path.exists(METADATA_FILE):
            return [], set()
        with open(METADATA_FILE, 'r') as file:
            metadata = [json.loads(line) for line in file]
        hash_set = {entry['hash'] for entry in metadata}
        return metadata, hash_set

    def _save_metadata(self):
        with open(METADATA_FILE, 'w') as file:
            for entry in self.metadata:
                json.dump(entry, file)
                file.write('\n')
        faiss.write_index(self.index, INDEX_FILE)

    def _generate_positive_hash(self, text):
        hash_object = hashlib.sha256(text.encode())
        return int(hash_object.hexdigest()[:16], 16)

    def add_texts(self, texts):
        embeddings = []
        new_entries = []
        for text in texts:
            text_hash = self._generate_positive_hash(text)
            if text_hash not in self.hash_set:
                embeddings.append(self.model.encode([text])[0])
                self.hash_set.add(text_hash)
                new_entries.append({"hash": text_hash, "text": text})
        if embeddings:
            self.index.add(np.array(embeddings))
            self.metadata.extend(new_entries)
            self._save_metadata()

    def search_vectors(self, query_text: str, top_k=5):
        if not hasattr(self, 'index') or self.index is None:
            raise ValueError("Vector index is not available.")

        query_embedding = self.model.encode([query_text])[0]
        query_np = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        extended_top_k = min(top_k * 3, len(self.metadata))
        distances, indices = self.index.search(query_np, extended_top_k)
        results = []
        seen_hashes = set()
        for idx in indices.flatten():
            if idx < len(self.metadata):
                metadata_entry = self.metadata[idx]
                text_hash = metadata_entry["hash"]
                text = metadata_entry["text"]
                if text_hash not in seen_hashes:
                    results.append({"id": text_hash, "rank": idx, "text": text})
                    seen_hashes.add(text_hash)
                if len(results) == top_k:
                    break
        return results

    def _process_text_and_add(self, text):
        sentences = re.split(r'(?<=[.!?]) +', text)
        for i in range(0, len(sentences), SENTENCE_CHUNK_SIZE):
            chunk_text = " ".join(sentences[i:i + SENTENCE_CHUNK_SIZE])
            self.add_texts([chunk_text])

    def add_pdf(self, pdf_path: str):
        with open(pdf_path, 'rb') as file:
            pdf = PdfReader(file)
            all_text = " ".join(page.extract_text() or "" for page in pdf.pages)
        self._process_text_and_add(all_text)

    def add_url(self, url: str):
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            self._process_text_and_add(text)
        else:
            print(f"Failed to fetch {url}: Status code {response.status_code}")
