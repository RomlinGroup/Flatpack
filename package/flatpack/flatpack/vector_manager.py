import datetime
import hashlib
import hnswlib
import json
import logging
import nltk
import numpy as np
import os
import requests
import time
import warnings

from bs4 import BeautifulSoup
from contextlib import redirect_stdout
from nltk.tokenize import sent_tokenize
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from typing import Dict, List

warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')


def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        with redirect_stdout(open(os.devnull, "w")):
            nltk.download('punkt', quiet=True)


download_nltk_data()

BATCH_SIZE = 64
EMBEDDINGS_FILE = "embeddings.npy"
INDEX_FILE = "hnsw_index.bin"
MAX_ELEMENTS = 10000
METADATA_FILE = "metadata.json"
SENTENCE_CHUNK_SIZE = 5
VECTOR_DIMENSION = 384


class VectorManager:
    def __init__(self, model_id='all-MiniLM-L6-v2', directory='./data'):
        self.directory = directory
        self.index_file = os.path.join(self.directory, INDEX_FILE)
        self.metadata_file = os.path.join(self.directory, METADATA_FILE)
        self.embeddings_file = os.path.join(self.directory, EMBEDDINGS_FILE)

        start_time = time.time()
        self.model = SentenceTransformer(model_id)
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")

        self.index = hnswlib.Index(space='cosine', dim=VECTOR_DIMENSION)
        self.metadata, self.hash_set, self.embeddings, self.ids = self._load_metadata_and_embeddings()
        self._initialize_index()

    def _initialize_index(self):
        if os.path.exists(self.index_file):
            self.index.load_index(self.index_file, max_elements=MAX_ELEMENTS)
        else:
            self.index.init_index(max_elements=MAX_ELEMENTS, ef_construction=200, M=16)
            if self.embeddings is not None and len(self.embeddings) > 0:
                self.index.add_items(self.embeddings, self.ids)
        self.index.set_ef(200)

    def is_index_ready(self):
        """Check if the index is ready for search operations."""
        return self.index.get_current_count() > 0

    def _load_metadata_and_embeddings(self):
        """Load metadata and embeddings from their respective files."""
        metadata = {}
        hash_set = set()
        embeddings = None
        ids = []

        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as file:
                metadata = json.load(file)
            hash_set = set(metadata.keys())

        if os.path.exists(self.embeddings_file):
            embeddings = np.load(self.embeddings_file)
            ids = list(map(int, metadata.keys()))

        return metadata, hash_set, embeddings, ids

    def _save_metadata_and_embeddings(self):
        """Save metadata and embeddings to their respective files."""
        with open(self.metadata_file, 'w') as file:
            json.dump(self.metadata, file, indent=2)
        np.save(self.embeddings_file, self.embeddings)
        self.index.save_index(self.index_file)

    @staticmethod
    def _generate_positive_hash(text):
        """Generate a positive hash for a given text."""
        hash_object = hashlib.sha256(text.encode())
        return int(hash_object.hexdigest()[:16], 16)

    def add_texts(self, texts, source_reference):
        """Add new texts and their embeddings to the index."""
        new_embeddings = []
        new_ids = []
        new_entries = {}

        for text in texts:
            text_hash = self._generate_positive_hash(text)
            if text_hash not in self.hash_set:
                embedding = self.model.encode([text])[0]
                new_embeddings.append(embedding)
                new_ids.append(text_hash)
                self.hash_set.add(text_hash)
                now = datetime.datetime.now()
                date_added = now.strftime("%Y-%m-%d %H:%M:%S")
                new_entries[text_hash] = {
                    "hash": text_hash,
                    "source": source_reference,
                    "date": date_added,
                    "text": text
                }

        if new_embeddings:
            new_embeddings = np.array(new_embeddings)
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack((self.embeddings, new_embeddings))

            self.index.add_items(new_embeddings, new_ids)
            self.metadata.update(new_entries)
            self._save_metadata_and_embeddings()

    def search_vectors(self, query_text: str, top_k=5):
        if not self.is_index_ready():
            logging.error("Index is not ready. No elements in the index.")
            return []

        start_time = time.time()
        query_embedding = self.model.encode([query_text])[0]
        print(f"Text encoded in {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        labels, distances = self.index.knn_query(query_embedding, k=top_k)
        print(f"Search completed in {time.time() - start_time:.2f} seconds")

        results = []
        for label, distance in zip(labels[0], distances[0]):
            entry = self.metadata.get(str(label))
            if entry:
                result = {
                    "id": entry['hash'],
                    "distance": float(distance),
                    "source": entry['source'],
                    "date": entry['date'],
                    "text": entry['text']
                }
                results.append(result)
                print(json.dumps(result, indent=2))

        return results

    def _process_text_and_add(self, text, source_reference):
        """Process the text into chunks and add to the index."""
        if not isinstance(text, str):
            logger.error("_process_text_and_add expected a string but got %s. Value: %s", type(text), text)
            return
        sentences = sent_tokenize(text)
        chunks = [" ".join(sentences[i:i + SENTENCE_CHUNK_SIZE]).strip() for i in
                  range(0, len(sentences), SENTENCE_CHUNK_SIZE)]
        self.add_texts(chunks, source_reference)

    def add_pdf(self, pdf_path: str):
        """Extract text from a PDF and add to the index."""
        if not os.path.isfile(pdf_path) or not pdf_path.lower().endswith('.pdf'):
            logging.error("Invalid PDF path provided: %s", pdf_path)
            return
        with open(pdf_path, 'rb') as file:
            pdf = PdfReader(file)
            all_text = " ".join(page.extract_text() or "" for page in pdf.pages)
        self._process_text_and_add(all_text, pdf_path)

    def add_url(self, url: str):
        """Fetch text from a URL and add to the index."""
        logging.info("Fetching URL: %s", url)
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            logging.debug("URL fetched successfully.")
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            logging.debug("Extracted text from HTML.")
            self._process_text_and_add(text, url)
            logging.info("Text added successfully.")
        except requests.RequestException as e:
            logging.error("Failed to fetch %s: %s", url, e)

    @staticmethod
    def get_wikipedia_text(page_title):
        """Fetch text from a Wikipedia page."""
        logger.info("Fetching Wikipedia page for: %s", page_title)
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

        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            page = next(iter(data["query"]["pages"].values()))
            return page.get("extract", "")
        except requests.RequestException as e:
            logging.error("Failed to fetch %s: %s", page_title, e)
            return ""

    def add_wikipedia_page(self, page_title):
        """Fetch and add a Wikipedia page to the index."""
        try:
            text = self.get_wikipedia_text(page_title)
            if text:
                logging.debug("Starting to process and add Wikipedia text.")
                self._process_text_and_add(text, f"wikipedia:{page_title}")
                logging.info("Wikipedia text added successfully.")
            else:
                logger.error("No text found for Wikipedia page: %s", page_title)
        except requests.RequestException as e:
            logger.error("Failed to fetch %s: %s", page_title, e)
