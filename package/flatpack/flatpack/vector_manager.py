import datetime
import faiss
import hashlib
import json
import logging
import nltk
import numpy as np
import os
import re
import requests

from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from typing import List
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

nltk.download('punkt')

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

    def is_index_ready(self):
        return self.index.ntotal > 0

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

    def add_texts(self, texts, source_reference):
        embeddings = []
        new_entries = []
        for text in texts:
            text_hash = self._generate_positive_hash(text)
            if text_hash not in self.hash_set:
                embeddings.append(self.model.encode([text])[0])
                self.hash_set.add(text_hash)
                now = datetime.datetime.now()
                date_added = now.strftime("%Y-%m-%d %H:%M:%S")
                new_entries.append({
                    "hash": text_hash,
                    "source": source_reference,
                    "date": date_added,
                    "text": text
                })
        if embeddings:
            self.index.add(np.array(embeddings))
            self.metadata.extend(new_entries)
            self._save_metadata()

    def search_vectors(self, query_text: str, top_k=5):
        if not hasattr(self, 'index') or self.index is None:
            raise ValueError("Vector index is not available.")

        # Encode the query text to get the embedding
        query_embedding = self.model.encode([query_text])[0]
        query_np = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        # Adjust the extended_top_k strategy based on duplication rates observed
        extended_top_k = min(top_k * 2, len(self.metadata))  # Adjust based on observed duplication rates
        distances, indices = self.index.search(query_np, extended_top_k)

        results = []
        seen_hashes = set()
        for distance, idx in zip(distances[0], indices[0]):
            if len(results) >= top_k:
                break  # Stop if we have enough results
            if idx >= len(self.metadata):
                continue  # Skip if index is out of bounds for metadata
            metadata_entry = self.metadata[idx]
            text_hash = metadata_entry["hash"]
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                # Now also append the source reference for each entry
                results.append({
                    "id": text_hash,
                    "distance": distance,
                    "source": metadata_entry["source"],
                    "date": metadata_entry["date"],
                    "text": metadata_entry["text"]
                })

        return results

    def _process_text_and_add(self, text, source_reference):
        if not isinstance(text, str):
            logging.error(f"_process_text_and_add expected a string but got {type(text)}. Value: {text}")
            return
        sentences = sent_tokenize(text)
        for i in range(0, len(sentences), SENTENCE_CHUNK_SIZE):
            chunk_text = " ".join(sentences[i:i + SENTENCE_CHUNK_SIZE]).strip()
            self.add_texts([chunk_text], source_reference)

    def add_pdf(self, pdf_path: str):
        with open(pdf_path, 'rb') as file:
            pdf = PdfReader(file)
            all_text = " ".join(page.extract_text() or "" for page in pdf.pages)
        self._process_text_and_add(all_text, pdf_path)

    def add_url(self, url: str):
        logging.info(f"Fetching URL: {url}")
        response = requests.get(url)
        if response.status_code == 200:
            logging.debug("URL fetched successfully.")
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            logging.debug("Extracted text from HTML.")
            self._process_text_and_add(text, url)  # Pass the URL as the source reference
            logging.info("Text added successfully.")
        else:
            logging.error(f"Failed to fetch {url}: Status code {response.status_code}")

    def get_wikipedia_text(self, page_title):
        """
        Fetches the plain text content of a Wikipedia page given its title using the Wikipedia API.
        """
        logging.info(f"Fetching Wikipedia page for: {page_title}")
        base_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "titles": page_title,
            "prop": "extracts",
            "explaintext": 1,
            "exsectionformat": "plain",
            "redirects": 1,
            "format": "json"
        }

        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raises HTTPError, if one occurred during the request

        data = response.json()
        page = next(iter(data["query"]["pages"].values()))
        return page.get("extract", "")

    def add_wikipedia_page(self, page_title):
        try:
            text = self.get_wikipedia_text(page_title)
            if text:
                logging.debug("Starting to process and add Wikipedia text.")
                self._process_text_and_add(text, f"wikipedia:{page_title}")
                logging.info("Wikipedia text added successfully.")
            else:
                logging.error(f"No text found for Wikipedia page: {page_title}")
        except requests.HTTPError as e:
            logging.error(f"Failed to fetch Wikipedia page: {page_title}. HTTPError: {e}")
