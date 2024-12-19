import datetime
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import warnings

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import hnswlib
import numpy as np
import requests

from bs4 import BeautifulSoup
from pypdf import PdfReader
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore', category=FutureWarning)

console = Console()

HOME_DIR = Path.home() / ".fpk"
HOME_DIR.mkdir(exist_ok=True)


def ensure_spacy_model():
    python_path = sys.executable
    pip_path = [python_path, "-m", "pip"]

    try:
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")

            if 'parser' in nlp.pipe_names:
                nlp.remove_pipe('parser')

            if 'sentencizer' not in nlp.pipe_names:
                nlp.add_pipe('sentencizer', config={
                    'punct_chars': None,
                    'overwrite': True
                })
            return nlp
        except OSError:
            pass
    except ImportError:
        pass

    console.print("")
    console.print("Installing spaCy (MIT) and downloading model...")

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
    ) as progress:
        install_task = progress.add_task("Installing spaCy...", total=None)

        try:
            process = subprocess.Popen(
                pip_path + ['install', 'spacy'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            for line in process.stdout:
                if "Installing collected packages" in line:
                    progress.update(install_task, description="Installing packages...")
                elif "Successfully installed" in line:
                    progress.update(install_task, description="SpaCy installed successfully!")
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, pip_path)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            console.print(f"[red]Error installing spaCy: {e}[/red]")
            return None

        download_task = progress.add_task("Downloading spaCy model...", total=None)

        try:
            process = subprocess.Popen(
                [python_path, '-m', 'spacy', 'download', 'en_core_web_sm'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )

            process.wait()

            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode,
                                                    [python_path, '-m', 'spacy', 'download', 'en_core_web_sm'])
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            console.print(f"[red]Error downloading spaCy model: {e}[/red]")
            return None

    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")

        if 'sentencizer' not in nlp.pipe_names:
            nlp.add_pipe('sentencizer', config={
                'punct_chars': None,
                'overwrite': True
            })
        return nlp
    except ImportError:
        console.print("[red]Failed to import spaCy after installation.[/red]")
        return None


nlp = ensure_spacy_model()

if nlp is None:
    console.print("[red]Failed to initialize spaCy. Please check your installation and try again.[/red]")
    sys.exit(1)

BATCH_SIZE = 32
EMBEDDINGS_FILE = "embeddings.npy"
INDEX_FILE = "hnsw_index.bin"
MAX_ELEMENTS = 5000
METADATA_FILE = "metadata.json"
SENTENCE_CHUNK_SIZE = 5
VECTOR_DIMENSION = 384


class VectorManager:
    def __init__(self, model_id='all-MiniLM-L6-v2', directory='./data'):
        self.directory = directory

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.index_file = os.path.join(self.directory, INDEX_FILE)
        self.metadata_file = os.path.join(self.directory, METADATA_FILE)
        self.embeddings_file = os.path.join(self.directory, EMBEDDINGS_FILE)

        (HOME_DIR / "cache").mkdir(parents=True, exist_ok=True)

        locks_dir = HOME_DIR / "cache" / ".locks"

        if locks_dir.exists():
            for lock_file in locks_dir.glob("**/*.lock"):
                try:
                    lock_file.unlink()
                except OSError:
                    pass

        self.model = model_id if isinstance(
            model_id,
            SentenceTransformer
        ) else SentenceTransformer(
            model_id,
            device='cpu',
            cache_folder=HOME_DIR / "cache"
        )

        self.index = hnswlib.Index(space='cosine', dim=VECTOR_DIMENSION)
        self.metadata, self.hash_set, self.embeddings, self.ids = self._load_metadata_and_embeddings()
        self._initialize_index()

        self.nlp = nlp

        if 'sentencizer' not in self.nlp.pipe_names:
            self.nlp.add_pipe('sentencizer', config={
                'punct_chars': None,
                'overwrite': True
            })

        gc.collect()

    def _initialize_index(self):
        if os.path.exists(self.index_file):
            self.index.load_index(self.index_file, max_elements=MAX_ELEMENTS)
        else:
            self.index.init_index(
                max_elements=MAX_ELEMENTS,
                ef_construction=200,
                M=64
            )

            if self.embeddings is not None and len(self.embeddings) > 0:
                batch_size = 1000
                for i in range(0, len(self.embeddings), batch_size):
                    batch_end = min(i + batch_size, len(self.embeddings))
                    self.index.add_items(
                        self.embeddings[i:batch_end],
                        self.ids[i:batch_end]
                    )
                    gc.collect()

        self.index.set_ef(100)

    def _load_metadata_and_embeddings(self):
        metadata = {}
        hash_set = set()
        embeddings = None
        ids = []

        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as file:
                chunk_size = 1024 * 1024
                buffer = ''

                while True:
                    chunk = file.read(chunk_size)
                    if not chunk:
                        break
                    buffer += chunk

                    try:
                        metadata.update(json.loads(buffer))
                        buffer = ''
                    except json.JSONDecodeError:
                        continue

                    hash_set.update(metadata.keys())

                    gc.collect()

        if os.path.exists(self.embeddings_file):
            embeddings = np.load(self.embeddings_file)
            ids = list(map(int, metadata.keys()))

        return metadata, hash_set, embeddings, ids

    def _save_metadata_and_embeddings(self):
        with open(self.metadata_file, 'w') as file:
            json.dump(self.metadata, file, indent=2)

        if self.embeddings is not None:
            np.save(self.embeddings_file, self.embeddings, allow_pickle=False)

        self.index.save_index(self.index_file)

    def is_index_ready(self):
        return self.index.get_current_count() > 0

    @staticmethod
    def _generate_positive_hash(text):
        """Generate a positive hash for a given text."""
        hash_object = hashlib.sha256(text.encode())
        return int(hash_object.hexdigest()[:16], 16)

    def add_texts(self, texts: List[str], source_reference: str):
        """Add new texts and their embeddings to the index."""
        batch_size = 32

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = []
            batch_ids = []
            batch_entries = {}

            hashes = [self._generate_positive_hash(text) for text in batch_texts]
            new_texts = [(h, text) for h, text in zip(hashes, batch_texts)
                         if h not in self.hash_set]

            if new_texts:
                batch_embeddings = self.model.encode(
                    [text for _, text in new_texts],
                    normalize_embeddings=True,
                    batch_size=len(new_texts)
                )

                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                for idx, (text_hash, text) in enumerate(new_texts):
                    self.hash_set.add(text_hash)
                    batch_ids.append(text_hash)
                    batch_entries[text_hash] = {
                        "hash": text_hash,
                        "source": source_reference,
                        "date": now,
                        "text": text
                    }

                if len(batch_embeddings) > 0:
                    batch_embeddings = np.array(batch_embeddings)
                    if self.embeddings is None:
                        self.embeddings = batch_embeddings
                    else:
                        self.embeddings = np.vstack((self.embeddings, batch_embeddings))

                    self.index.add_items(batch_embeddings, batch_ids)
                    self.metadata.update(batch_entries)

                    self._save_metadata_and_embeddings()

            del batch_embeddings, batch_ids, batch_entries
            gc.collect()

    def search_vectors(self, query: str, top_k: int = 10, recency_weight: float = 0.5) -> List[Dict[str, Any]]:
        """Search vectors with an optional bias toward recent results."""
        if not self.is_index_ready():
            return []

        query = query.strip()
        if not query:
            return []

        try:
            query_embedding = self.model.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False
            )

            actual_k = min(top_k, self.index.get_current_count())
            if actual_k < 1:
                return []

            labels, distances = self.index.knn_query(query_embedding, k=actual_k)

            if len(labels) == 0 or len(labels[0]) == 0:
                return []

            results = []
            now = datetime.datetime.now()

            for idx, distance in zip(labels[0], distances[0]):
                str_idx = str(idx)
                if str_idx in self.metadata:
                    meta = self.metadata[str_idx]
                    doc_date = datetime.datetime.strptime(meta['date'], "%Y-%m-%d %H:%M:%S")
                    recency_score = 1 / (1 + (now - doc_date).total_seconds() / 86400)
                    combined_score = recency_weight * recency_score + (1 - recency_weight) * (1 - distance)

                    results.append({
                        'id': int(idx),
                        'text': meta['text'],
                        'source': meta['source'],
                        'distance': float(distance),
                        'recency_score': recency_score,
                        'combined_score': combined_score
                    })

            results = sorted(results, key=lambda x: x['combined_score'], reverse=True)
            return results[:top_k]
        except Exception:
            return []

    @staticmethod
    def _preprocess_text(text: str) -> str:
        """Clean and normalize text before processing."""
        text = ' '.join(text.split())

        text = text.replace('•', '.')
        text = text.replace('…', '...')
        text = text.replace('\n', ' ')

        return text.strip()

    def _process_text_and_add(self, text: str, source_reference: str):
        """Process text into semantic chunks using improved sentence segmentation."""
        if not isinstance(text, str) or not text.strip():
            return

        text = self._preprocess_text(text)
        doc = self.nlp(text)

        sentences = []
        current_chunk = []

        for sent in doc.sents:
            clean_sent = sent.text.strip()

            if not clean_sent or len(clean_sent.split()) < 3:
                continue

            if (clean_sent.endswith(('.', '!', '?', '"', "'", ')', ']', '}')) or
                    clean_sent.endswith((':", "."', '."', '!"', '?"'))):

                current_chunk.append(clean_sent)

                if len(current_chunk) >= SENTENCE_CHUNK_SIZE:
                    chunk_text = ' '.join(current_chunk)
                    if len(chunk_text.split()) >= 10:
                        self.add_texts([chunk_text], source_reference)
                    current_chunk = []
            else:
                if current_chunk:
                    current_chunk[-1] = current_chunk[-1] + ' ' + clean_sent
                else:
                    current_chunk.append(clean_sent)

        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text.split()) >= 10:
                self.add_texts([chunk_text], source_reference)

        del doc
        gc.collect()

    def add_pdf(self, pdf_path: str):
        """Extract text from PDF with improved text handling."""
        if not os.path.isfile(pdf_path) or not pdf_path.lower().endswith('.pdf'):
            return

        with open(pdf_path, 'rb') as file:
            pdf = PdfReader(file)

            page_texts = []
            current_text_length = 0

            for page in pdf.pages:
                text = page.extract_text() or ""
                if text:
                    page_texts.append(text)
                    current_text_length += len(text)

                    if current_text_length >= 10000:
                        combined_text = ' '.join(page_texts)
                        self._process_text_and_add(combined_text, pdf_path)
                        page_texts = []
                        current_text_length = 0
                        gc.collect()

            if page_texts:
                combined_text = ' '.join(page_texts)
                self._process_text_and_add(combined_text, pdf_path)

    def add_url(self, url: str):
        """Fetch text from a URL and add to the index."""
        try:
            with requests.get(url, timeout=10, stream=True) as response:
                response.raise_for_status()
                content = b''
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        content += chunk
                        if len(content) >= 1_000_000:
                            soup = BeautifulSoup(content, 'html.parser')
                            text = soup.get_text(separator=' ', strip=True)
                            self._process_text_and_add(text, url)
                            content = b''
                            del soup, text
                            gc.collect()

                if content:
                    soup = BeautifulSoup(content, 'html.parser')
                    text = soup.get_text(separator=' ', strip=True)
                    self._process_text_and_add(text, url)
        except requests.RequestException:
            pass

    @staticmethod
    def get_wikipedia_text(page_title):
        """Fetch text from a Wikipedia page."""
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
        except requests.RequestException:
            return ""

    def add_wikipedia_page(self, page_title):
        """Fetch and add a Wikipedia page to the index."""
        try:
            text = self.get_wikipedia_text(page_title)
            if text:
                self._process_text_and_add(text, f"wikipedia:{page_title}")
            else:
                pass
        except requests.RequestException:
            pass
