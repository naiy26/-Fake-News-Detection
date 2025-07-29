"""
Fake News Detection System using Spark LLM with RAG
==================================================
This system combines:
1. Web search for retrieving relevant articles
2. Embedding-based similarity for pre-filtering
3. FAISS for efficient similarity search
4. Spark LLM for fact-checking and verdict generation
5. Sentiment analysis to detect emotional manipulation
6. Named Entity Recognition to extract key entities
"""

# Required imports
import os
import json
import base64
import hashlib
import hmac
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime,UTC
from typing import List, Dict, Tuple, Optional, Any, Union
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import faiss
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import spacy
import logging
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import re
from functools import lru_cache
import time
from cachetools import TTLCache


nltk.download('punkt_tab')
# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('punkt')
    nltk.download('vader_lexicon')


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("truthlens.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("SpaCy model loaded successfully")
except:
    logger.warning("SpaCy model not found. Downloading...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Load environment variables (still keeping this for other credentials)
load_dotenv()

# API Credentials
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_CX = os.getenv('GOOGLE_CX')

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

SPARK_API_PASSWORD = os.getenv('SPARK_API_PASSWORD')
llm_cache = TTLCache(maxsize=1000, ttl=3600)

# Validate credentials
def validate_credentials():
    """Validate that all required API credentials are present"""
    # Check for Google credentials
    missing_google = []
    if not GOOGLE_API_KEY:
        missing_google.append('Google Search API Key')
    if not GOOGLE_CX:
        missing_google.append('Google Custom Search Engine ID')
    
    if missing_google:
        print(f"Warning: Missing Google credentials: {', '.join(missing_google)}")
        print("Web search functionality will be limited.")
    
    # Check Spark API credentials
    if not SPARK_API_PASSWORD:
        print("Warning: Missing Spark API Password. Using default value.")
    else:
        print(f"✓ Spark API credentials are configured with password: {SPARK_API_PASSWORD[:5]}... (length: {len(SPARK_API_PASSWORD)})")
    
    return True

result_cache = {}

# Web search component
class WebSearchEngine:
    """Component for retrieving articles from the web using Google Search API"""
    
    def __init__(self, api_key: str, cx: str):
        """Initialize the search engine with API credentials
        
        Args:
            api_key: Google API Key
            cx: Google Custom Search Engine ID
        """
        self.api_key = api_key
        self.cx = cx
        self.search_url = "https://www.googleapis.com/customsearch/v1"
    
    def search(self, query: str, num_results: int = 10) -> List[Dict[str, str]]:
        """Search for articles related to a query
        
        Args:
            query: The search query
            num_results: Number of results to retrieve (max 10 for free tier)
            
        Returns:
            List of dictionaries containing article information
        """
        cache_key = f"search_{hash(query)}_{num_results}"
        if cache_key in result_cache:
            logger.info(f"Returning cached search results for: {query}")
            return result_cache[cache_key]
        
        params = {
            "key": self.api_key,
            "cx": self.cx,
            "q": query,
            "num": min(num_results, 10)  # Max 10 for free tier
        }
        
        try:
            logger.info(f"Searching for: {query}")
            response = requests.get(self.search_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract relevant information
            articles = []
            for item in data.get("items", []):
                article = {
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "link": item.get("link", ""),
                    "source": item.get("displayLink", ""),
                    "published_date": item.get("pagemap", {}).get("metatags", [{}])[0].get("article:published_time", "")
                }
                
                # Create a full text field combining title and snippet
                article["full_text"] = f"{article['title']} {article['snippet']}"
                articles.append(article)
            
            # Cache the results
            result_cache[cache_key] = articles
            logger.info(f"Found {len(articles)} articles for: {query}")
            return articles
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error in web search: {e}")
            logger.debug("Full API Response: " + response.text)
            return []
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return []
    
    def fetch_article_content(self, url: str) -> Optional[str]:
        """Fetch and extract the main content from an article
        
        Args:
            url: URL of the article
            
        Returns:
            Extracted article content or None if failed
        """
        # Check cache first
        cache_key = f"content_{hash(url)}"
        if cache_key in result_cache:
            return result_cache[cache_key]
        
        try:
            logger.info(f"Fetching content from: {url}")
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Use BeautifulSoup for more sophisticated content extraction
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove non-content elements
            for element in soup(["script", "style", "header", "footer", "nav"]):
                element.decompose()
            
            # Extract paragraphs with reasonable length
            paragraphs = []
            for p in soup.find_all('p'):
                text = p.get_text().strip()
                if len(text) > 40:
                    paragraphs.append(text)
            
            if not paragraphs:
                # Fallback to extracting text from the body
                body_text = soup.body.get_text() if soup.body else ""
                content = " ".join(body_text.split())
            else:
                content = " ".join(paragraphs)
            
            # Truncate to a reasonable length
            content = content[:5000]
            
            # Cache the result
            result_cache[cache_key] = content
            return content
            
        except Exception as e:
            logger.error(f"Error fetching article content: {e}")
            return None


# URL handling component
class URLHandler:
    """Component for handling URL inputs and extracting article content"""
    
    def __init__(self):
        """Initialize the URL handler"""
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    
    def is_url(self, text: str) -> bool:
        """Check if the input text is a URL
        
        Args:
            text: Text to check
            
        Returns:
            Boolean indicating if text is a URL
        """
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ipv4
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return bool(url_pattern.match(text))
    
    def extract_article(self, url: str) -> Dict[str, Any]:
        """Extract article content and metadata from a URL with improved handling for major news sites
        
        Args:
            url: URL to extract from
            
        Returns:
            Dictionary with article content and metadata
        """
        try:
            logger.info(f"Extracting article from URL: {url}")
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            from bs4 import BeautifulSoup
            import trafilatura
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Identify the news source from the URL
            news_source = self._identify_news_source(url)
            extracted_text = None
            
            # Site-specific content extraction
            if news_source == 'cnn':
                article_body = soup.find('div', class_='article__content')
                if article_body:
                    paragraphs = article_body.find_all('p')
                    extracted_text = ' '.join([p.get_text().strip() for p in paragraphs])
                    
            elif news_source == 'bbc':
                article_body = soup.find('article')
                if article_body:
                    paragraphs = article_body.find_all('p')
                    extracted_text = ' '.join([p.get_text().strip() for p in paragraphs])
                
            elif news_source == 'nytimes':
                article_sections = soup.find_all('section', class_='meteredContent')
                if article_sections:
                    paragraphs = []
                    for section in article_sections:
                        paragraphs.extend(section.find_all('p'))
                    extracted_text = ' '.join([p.get_text().strip() for p in paragraphs])
            
            # If site-specific extraction failed or it's not a recognized site, try trafilatura
            if not extracted_text:
                extracted_text = trafilatura.extract(response.text)
            
            # Fallback to BeautifulSoup if trafilatura fails
            if not extracted_text:
                # Remove non-content elements
                for element in soup(["script", "style", "header", "footer", "nav"]):
                    element.decompose()
                
                # Extract paragraphs with reasonable length
                paragraphs = []
                for p in soup.find_all('p'):
                    text = p.get_text().strip()
                    if len(text) > 40:
                        paragraphs.append(text)
                
                if not paragraphs:
                    # Fallback to extracting text from the body
                    body_text = soup.body.get_text() if soup.body else ""
                    extracted_text = " ".join(body_text.split())
                else:
                    extracted_text = " ".join(paragraphs)
            
            # Extract metadata
            metadata = self._extract_metadata(soup, url, news_source)
            
            # Build result
            result = {
                "content": extracted_text[:10000],  # Limit content length
                "url": url,
                "title": metadata.get("title", "Unknown Title"),
                "author": metadata.get("author", "Unknown Author"),
                "published_date": metadata.get("published_date", "Unknown Date"),
                "source_domain": self._extract_domain(url),
                "is_url_input": True
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting article from URL: {e}")
            return {
                "content": "",
                "url": url,
                "title": "Failed to extract",
                "author": "Unknown",
                "published_date": "Unknown",
                "source_domain": self._extract_domain(url),
                "error": str(e),
                "is_url_input": True
            }

    def _identify_news_source(self, url: str) -> str:
        """Identify the news source from a URL
        
        Args:
            url: The URL to analyze
            
        Returns:
            String identifying the news source
        """
        url_lower = url.lower()
        
        if 'cnn.com' in url_lower:
            return 'cnn'
        elif 'bbc.com' in url_lower or 'bbc.co.uk' in url_lower:
            return 'bbc'
        elif 'nytimes.com' in url_lower:
            return 'nytimes'
        elif 'washingtonpost.com' in url_lower:
            return 'washingtonpost'
        elif 'theguardian.com' in url_lower:
            return 'guardian'
        elif 'reuters.com' in url_lower:
            return 'reuters'
        elif 'apnews.com' in url_lower:
            return 'ap'
        elif 'foxnews.com' in url_lower:
            return 'fox'
        elif 'nbcnews.com' in url_lower:
            return 'nbc'
        elif 'abcnews.go.com' in url_lower:
            return 'abc'
        elif 'cbsnews.com' in url_lower:
            return 'cbs'
        else:
            return 'unknown'
    
   
    def _extract_metadata(self, soup, url: str, news_source: str = 'unknown') -> Dict[str, str]:
        """Extract metadata from HTML with improved handling for major news sites
        
        Args:
            soup: BeautifulSoup object
            url: Original URL
            news_source: Identified news source
            
        Returns:
            Dictionary with metadata
        """
        metadata = {
            "title": "Unknown Title",
            "author": "Unknown Author",
            "published_date": "Unknown Date"
        }
        
        # Site-specific title extraction
        if news_source == 'cnn':
            headline_tag = soup.find('h1', class_='headline__text')
            if headline_tag:
                metadata["title"] = headline_tag.get_text().strip()
        
        elif news_source == 'bbc':
            headline_tag = soup.find('h1', id='main-heading')
            if headline_tag:
                metadata["title"] = headline_tag.get_text().strip()
        
        elif news_source == 'nytimes':
            headline_tag = soup.find('h1', class_='css-1l24qy5')
            if headline_tag:
                metadata["title"] = headline_tag.get_text().strip()
        
        elif news_source == 'washingtonpost':
            headline_tag = soup.find('h1', class_='font-md')
            if headline_tag:
                metadata["title"] = headline_tag.get_text().strip()
        
        elif news_source == 'guardian':
            headline_tag = soup.find('h1', class_='dcr-125vfar')
            if headline_tag:
                metadata["title"] = headline_tag.get_text().strip()
        
        # Try OG tags for title (many sites use these)
        if metadata["title"] == "Unknown Title":
            og_title = soup.find('meta', property='og:title')
            if og_title and og_title.get('content'):
                metadata["title"] = og_title.get('content').strip()
        
        # Fallback to standard title tag
        if metadata["title"] == "Unknown Title" and soup.title:
            metadata["title"] = soup.title.get_text().strip()
        
        # Site-specific author extraction
        if news_source == 'cnn':
            cnn_authors = []
            byline_tags = soup.find_all('div', class_='byline__names')
            for byline in byline_tags:
                author_links = byline.find_all('a', class_='byline__name')
                for author_link in author_links:
                    cnn_authors.append(author_link.get_text().strip())
            
            if cnn_authors:
                metadata["author"] = ", ".join(cnn_authors)
        
        elif news_source == 'bbc':
            byline_tag = soup.find('div', class_='ssrcss-68pt20-Text-TextContributorName')
            if byline_tag:
                metadata["author"] = byline_tag.get_text().strip()
        
        elif news_source == 'nytimes':
            byline_tags = soup.find_all('span', class_='css-1baulvz')
            if byline_tags:
                authors = [tag.get_text().strip() for tag in byline_tags if tag.get_text().strip()]
                if authors:
                    metadata["author"] = ", ".join(authors)
                    
        elif news_source == 'washingtonpost':
            author_tag = soup.find('a', class_='css-nowh2b')
            if author_tag:
                metadata["author"] = author_tag.get_text().strip()
        
        # Try standard meta tags for author
        if metadata["author"] == "Unknown Author":
            author_meta_tags = [
                soup.find('meta', attrs={'name': 'author'}),
                soup.find('meta', attrs={'property': 'article:author'}),
                soup.find('meta', attrs={'property': 'og:author'})
            ]
            
            for tag in author_meta_tags:
                if tag and tag.get('content'):
                    metadata["author"] = tag.get('content').strip()
                    break
        
        # Try common author class/ID patterns
        if metadata["author"] == "Unknown Author":
            author_patterns = [
                soup.find(class_=['author', 'byline', 'byline__name', 'writer', 'creator']),
                soup.find(id=['author', 'byline']),
                soup.find('a', class_=['author', 'byline']),
                soup.find('span', class_=['author', 'byline'])
            ]
            
            for pattern in author_patterns:
                if pattern:
                    author_text = pattern.get_text().strip()
                    if author_text and len(author_text) < 100:  # Avoid getting paragraphs
                        metadata["author"] = author_text
                        break
        
        # Site-specific date extraction
        if news_source == 'cnn':
            date_tag = soup.find('div', class_='timestamp')
            if date_tag:
                metadata["published_date"] = date_tag.get_text().strip()
        
        elif news_source == 'bbc':
            time_tag = soup.find('time')
            if time_tag and time_tag.has_attr('datetime'):
                metadata["published_date"] = time_tag['datetime']
        
        elif news_source == 'nytimes':
            time_tag = soup.find('time')
            if time_tag and time_tag.has_attr('datetime'):
                metadata["published_date"] = time_tag['datetime']
        
        # Try standard meta tags for date
        if metadata["published_date"] == "Unknown Date":
            date_meta_tags = [
                soup.find('meta', attrs={'property': 'article:published_time'}),
                soup.find('meta', attrs={'name': 'publication_date'}),
                soup.find('meta', attrs={'name': 'date'}),
                soup.find('meta', attrs={'property': 'og:published_time'}),
                soup.find('meta', attrs={'itemprop': 'datePublished'})
            ]
            
            for tag in date_meta_tags:
                if tag and tag.get('content'):
                    metadata["published_date"] = tag.get('content').strip()
                    break
        
        # Try common date class/ID patterns
        if metadata["published_date"] == "Unknown Date":
            date_patterns = [
                soup.find(class_=['date', 'published', 'timestamp', 'article-date']),
                soup.find(id=['date', 'published-date', 'publication-date']),
                soup.find('time')
            ]
            
            for pattern in date_patterns:
                if pattern:
                    if pattern.get('datetime'):
                        metadata["published_date"] = pattern.get('datetime')
                    else:
                        date_text = pattern.get_text().strip()
                        if date_text and len(date_text) < 100:  # Avoid getting paragraphs
                            metadata["published_date"] = date_text
                    break
        
        return metadata
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL
        
        Args:
            url: URL
            
        Returns:
            Domain name
        """
        try:
            from urllib.parse import urlparse
            parsed_uri = urlparse(url)
            domain = '{uri.netloc}'.format(uri=parsed_uri)
            return domain
        except:
            return "unknown_domain"



# Embedding and similarity component
class EmbeddingEngine:
    """Component for creating and comparing text embeddings"""
    
    def __init__(self, model_name: str = "distilbert-base-nli-mean-tokens"):
        """Initialize with a sentence transformer model
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        try:
            self.model = SentenceTransformer(model_name)
            print(f"✓ Model '{model_name}' loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Error loading model {model_name}: {e}")
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Array of embeddings
        """
        if not texts:
            return np.array([])
            
        return self.model.encode(texts)
    
    def compute_similarity(self, query_embedding: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and documents
        
        Args:
            query_embedding: Embedding of the query
            document_embeddings: Embeddings of the documents
            
        Returns:
            Array of similarity scores
        """
        return cosine_similarity(query_embedding.reshape(1, -1), document_embeddings)[0]
    
    def filter_by_threshold(self, texts: List[str], similarities: np.ndarray, threshold: float = 0.70) -> List[str]:
        """Filter texts based on similarity threshold
        
        Args:
            texts: List of text strings
            similarities: Array of similarity scores
            threshold: Minimum similarity score to keep
            
        Returns:
            Filtered list of texts
        """
        filtered_texts = []
        for text_dict, sim in zip(texts, similarities):
            if sim >= threshold:
                text_dict_copy = text_dict.copy()
                text_dict_copy["similarity"] = float(sim)
                filtered_texts.append(text_dict_copy)
        
        # Sort by similarity score (descending)
        filtered_texts.sort(key=lambda x: x["similarity"], reverse=True)
        return filtered_texts
    
    
# FAISS index for efficient similarity search
class FAISSIndexer:
    """Component for efficient similarity search using FAISS"""
    
    def __init__(self):
        """Initialize the FAISS indexer"""
        self.index = None
        self.dimension = None
    
    def create_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create a FAISS index from embeddings
        
        Args:
            embeddings: Document embeddings
            
        Returns:
            FAISS index
        """
        if embeddings is None or embeddings.size == 0:
            raise ValueError("No embeddings provided for FAISS index")
            
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        return self.index
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar documents
        
        Args:
            query_embedding: Embedding of the query
            k: Number of results to return
            
        Returns:
            Tuple of (distances, indices)
        """
        if self.index is None:
            raise ValueError("FAISS index not created yet")
            
        return self.index.search(query_embedding, k)
    
    def retrieve_docs(self, query_embedding: np.ndarray, articles: List[Dict[str, str]], k: int = 5) -> List[Dict[str, str]]:
        """Retrieve similar documents for a query
        
        Args:
            query_embedding: Embedding of the query
            articles: List of article dictionaries
            k: Number of results to return
            
        Returns:
            List of retrieved articles
        """
        distances, indices = self.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(articles):
                article = articles[idx].copy()
                # Convert L2 distance to a similarity score (smaller distance = higher similarity)
                max_distance = 10.0  # Arbitrary max distance for normalization
                similarity = max(0, 1.0 - distances[0][i] / max_distance)
                article["similarity"] = float(similarity)
                results.append(article)
        
        return results


# Named Entity Recognition component
class EntityExtractor:
    """Component for extracting named entities from text"""
    
    def __init__(self):
        """Initialize the entity extractor"""
        self.nlp = nlp
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dictionary of entity types and their mentions
        """
        doc = self.nlp(text)
        
        entities = {}
        for ent in doc.ents:
            entity_type = ent.label_
            entity_text = ent.text
            
            if entity_type not in entities:
                entities[entity_type] = []
            
            if entity_text not in entities[entity_type]:
                entities[entity_type].append(entity_text)
        
        return entities
    
    def get_key_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract and filter for key entity types
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dictionary of key entity types and their mentions
        """
        all_entities = self.extract_entities(text)
        
        # Filter for key entity types
        key_entity_types = ["PERSON", "ORG", "GPE", "DATE", "PERCENT", "MONEY", "QUANTITY"]
        key_entities = {k: v for k, v in all_entities.items() if k in key_entity_types}
        
        return key_entities



# Knowledge Graph Component (simplified version)
class KnowledgeGraph:
    """Component for entity verification using a knowledge graph"""
    
    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        """Initialize the knowledge graph component
        
        Args:
            uri: Neo4j URI
            user: Neo4j username
            password: Neo4j password
        """
        self.uri = uri or NEO4J_URI
        self.user = user or NEO4J_USER
        self.password = password or NEO4J_PASSWORD
        self.available = False
        
        # Only attempt to connect if credentials are provided
        if self.uri and self.user and self.password:
            try:
                from neo4j import GraphDatabase
                self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                self.available = True
                logger.info("Neo4j connection established successfully")
            except Exception as e:
                logger.warning(f"Neo4j connection failed: {e}")
                self.driver = None
        else:
            logger.warning("Neo4j credentials not provided, knowledge graph functionality will be limited")
            self.driver = None
    
    def verify_entity_relationships(self, entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """Verify relationships between entities
        
        Args:
            entities: Dictionary of entity types and their mentions
            
        Returns:
            Dictionary with verification results
        """
        if not self.available or not self.driver:
            return {"status": "unavailable", "message": "Knowledge graph not available"}
        
        verification_results = {"status": "success", "verified": [], "unverified": []}
        
        try:
            with self.driver.session() as session:
                # For each entity, check if it exists in the graph
                for entity_type, entity_mentions in entities.items():
                    for entity in entity_mentions:
                        # Cypher query to find the entity
                        query = f"""
                        MATCH (e) 
                        WHERE e.name = $entity_name OR e.alias = $entity_name
                        RETURN e
                        """
                        
                        result = session.run(query, entity_name=entity)
                        records = list(result)
                        
                        if records:
                            verification_results["verified"].append({
                                "entity": entity,
                                "type": entity_type,
                                "found_in_kg": True
                            })
                        else:
                            verification_results["unverified"].append({
                                "entity": entity,
                                "type": entity_type,
                                "found_in_kg": False
                            })
                
                return verification_results
                
        except Exception as e:
            logger.error(f"Error verifying entities: {e}")
            return {"status": "error", "message": str(e)}
    
    def close(self):
        """Close the Neo4j connection"""
        if self.available and self.driver:
            self.driver.close()


# Spark LLM component (replacing Spark LLM)
class SparkLLMFactChecker:
    """Component for fact-checking using Spark LLM API"""
   
    
    def __init__(self, api_password: str = None):
        """Initialize with Spark API password
        
        Args:
            api_password: Spark API Password (Format: "APIPassword" from console)
        """
        # Load from environment if not provided
        self.api_password = api_password or os.getenv('SPARK_API_PASSWORD', "cmdYpTbmpTgclKqTfhCE:ZOkmMTpDiqULISaijjgd")
        
        # HTTP API URL from the documentation
        self.api_url = "https://spark-api-open.xf-yun.com/v1/chat/completions"
        
        print(f"✓ Spark LLM API configured with password: {self.api_password[:5]}... (length: {len(self.api_password)})")
    
    def preprocess_evidence(self, retrieved_articles: List[Dict[str, str]]) -> str:
        """Optimized evidence preprocessing to reduce token count
        
        Args:
            retrieved_articles: List of retrieved articles
            
        Returns:
            Preprocessed evidence text
        """
        evidence_text = ""
        # Use fewer articles (2 max) and shorter snippets
        for i, article in enumerate(retrieved_articles[:2], 1):
            title = article.get("title", "")[:100]  # Truncate long titles
            snippet = article.get("snippet", "")[:150]  # Shorter snippets
            source = article.get("source", "")[:30]  # Shorter source names
            
            evidence_text += f"[{i}] {title} | {source} | {snippet}\n"
        
        # Truncate if still too long
        evidence_text = evidence_text[:1500]  # Reduced from 2500
        return evidence_text

    def get_cache_key(self, text, model="4.0Ultra"):
        """Generate a deterministic cache key for a prompt and model
        
        Args:
            text: The prompt text
            model: The model name
            
        Returns:
            String cache key
        """
        # Use a hash function to create a compact, unique key
        import hashlib
        text_bytes = text.encode('utf-8')
        hash_obj = hashlib.md5(text_bytes)
        return f"{model}_{hash_obj.hexdigest()}"
    
    def query_with_cache(self, prompt, model="4.0Ultra", temperature=0.2, max_tokens=1000):
        """Query the Spark LLM with caching
        
        Args:
            prompt: The prompt to send
            model: Model name
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLM response
        """
        cache_key = self.get_cache_key(prompt, model)
        
        # Check if result is in cache
        if cache_key in llm_cache:
            logger.info(f"Cache hit for prompt: {prompt[:50]}...")
            return llm_cache[cache_key]
        
        # Make the API request
        try:
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional fact-checker with expertise in detecting misinformation."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_password}"
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            if result.get("code") == 0:
                content = result["choices"][0]["message"]["content"]
                # Store in cache
                llm_cache[cache_key] = content
                return content
            else:
                logger.error(f"API Error {result.get('code')}: {result.get('message')}")
                return None
                
        except Exception as e:
            logger.error(f"API Error: {str(e)}")
            return None
    
    def build_optimized_prompt(self, claim: str, evidence_text: str, entities: Dict[str, List[str]] = None, source_credibility: float = None) -> str:
        """
        Build an optimized prompt for accurate and concise fact-checking.

        Args:
            claim: The claim to verify.
            evidence_text: Supporting evidence for evaluation.
            entities: Named entities extracted from the claim (optional).
            source_credibility: Credibility score of the source, if available.

        Returns:
            A refined prompt string to guide the language model in English.
        """

        # Format key entities for context
        entity_context = ""
        if entities:
            key_types = {"PERSON", "ORG", "GPE", "DATE"}
            filtered_entities = {k: v for k, v in entities.items() if k in key_types and v}
            if filtered_entities:
                entity_parts = [f"{etype}: {', '.join(entities[etype][:3])}" for etype in filtered_entities]
                entity_context = "Entities: " + "; ".join(entity_parts)

        # Format source credibility
        credibility_context = ""
        if source_credibility is not None:
            credibility_level = (
                "HIGH" if source_credibility > 0.7 else 
                "MEDIUM" if source_credibility > 0.4 else 
                "LOW"
            )
            credibility_context = f"Source Credibility: {credibility_level}"

        # Combine additional context if available
        additional_context = "\n".join(filter(None, [entity_context, credibility_context]))

        few_shot_examples = """
    Example 1:
    Claim: "Scientists have confirmed that drinking lemon water cures cancer."
    Evidence: Research shows lemon has some antioxidant properties but no studies demonstrate it cures cancer. Medical authorities confirm there is no evidence for this claim.
    Verdict: False
    Confidence: 95%
    Reasoning: This claim contradicts established medical science. While lemons have health benefits, no peer-reviewed studies support anticancer properties of the magnitude claimed.
    """


        # Final optimized prompt
        prompt = f"""You are an expert fact-checking assistant. Analyze the claim based on the given evidence and provide your response in ENGLISH.

    Claim:
    "{claim}"

    Evidence:
    {evidence_text}

    {additional_context}
    {few_shot_examples}

    Respond using this format:
    Verdict: [True / Partially True / False / Unverified]
    Confidence: [0-100%]
    Explanation: [Brief explanation based on the evidence]
    """
        return prompt

    def batch_process_requests(self, prompts, model="4.0Ultra", temperature=0.2, max_tokens=1000):
        """Process multiple requests in a single batch to reduce API call overhead
        
        Args:
            prompts: List of prompts to process
            model: Model name
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            List of responses
        """
        try:
            # Create a batch request
            batch_payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional fact-checker with expertise in detecting misinformation."
                    }
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            # Add all prompts as separate messages
            for i, prompt in enumerate(prompts):
                batch_payload["messages"].append({
                    "role": "user",
                    "content": prompt
                })
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_password}"
            }
            
            # Make a single API call for all prompts
            response = requests.post(self.api_url, headers=headers, json=batch_payload)
            response.raise_for_status()
            result = response.json()
            
            # Extract and return all responses
            responses = []
            if result.get("code") == 0:
                for choice in result["choices"]:
                    responses.append(choice["message"]["content"])
                return responses
            else:
                logger.error(f"Batch API Error {result.get('code')}: {result.get('message')}")
                return [None] * len(prompts)
                
        except Exception as e:
            logger.error(f"Batch API Error: {str(e)}")
            return [None] * len(prompts)
    
    
    
    def analyze_claim_evidence_alignment(self, claim: str, evidence: str) -> Dict[str, Any]:
        """Analyze alignment between claim and evidence
        
        Args:
            claim: The claim to analyze
            evidence: Evidence text
            
        Returns:
            Dictionary with alignment analysis
        """
        # Extract entities from claim and evidence using simple method
        def extract_entities(text: str) -> List[str]:
            import re
            from collections import Counter
            
            # Extract capitalized words that might be entities
            words = re.findall(r'\b[A-Z][a-zA-Z]*\b', text)
            
            # Count occurrences
            word_counts = Counter(words)
            
            # Return the most common entities (occurring more than once)
            return [word for word, count in word_counts.most_common() if count > 1]
        
        # Extract entities from claim and evidence
        claim_entities = set(extract_entities(claim))
        evidence_entities = set(extract_entities(evidence))
        
        # Check entity overlap
        common_entities = claim_entities.intersection(evidence_entities)
        entity_coverage = len(common_entities) / len(claim_entities) if claim_entities else 0
        
        # Get claim and evidence embeddings for semantic similarity
        embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Lightweight model for this purpose
        claim_embedding = embedding_model.encode([claim])[0]
        evidence_embedding = embedding_model.encode([evidence])[0]
        
        # Calculate cosine similarity
        similarity = cosine_similarity(
            claim_embedding.reshape(1, -1), 
            evidence_embedding.reshape(1, -1)
        )[0][0]
        
        # Determine confidence based on alignment factors
        confidence = (entity_coverage * 0.5) + (similarity * 0.5)
        confidence = min(max(confidence, 0.1), 0.95)  # Ensure reasonable bounds
        
        return {
            "entity_overlap": {
                "claim_entities": list(claim_entities),
                "evidence_entities": list(evidence_entities),
                "common_entities": list(common_entities),
                "entity_coverage": entity_coverage
            },
            "semantic_similarity": float(similarity),
            "confidence": float(confidence)
        }
    
    def fact_check(self, claim: str, retrieved_articles: List[Dict[str, str]], entities: Dict[str, List[str]] = None, source_credibility: float = None) -> Optional[str]:
        """Optimized fact-checking using Spark LLM
        
        Args:
            claim: The claim to fact-check
            retrieved_articles: List of retrieved articles
            entities: Optional dict of extracted entities
            source_credibility: Optional credibility score of the source
            
        Returns:
            Fact-checking result dictionary
        """
        # Check cache first
        cache_key = f"fact_check_{hash(claim)}_{hash(str(retrieved_articles))}"
        if cache_key in result_cache:
            logger.info(f"Returning cached fact-check result for: {claim}")
            return result_cache[cache_key]
        
        # Preprocess evidence - limit to top 2 articles instead of 3 to reduce token count
        evidence = self.preprocess_evidence(retrieved_articles[:2])
        
        # Build a more concise prompt
        prompt = self.build_optimized_prompt(claim, evidence, entities, source_credibility)
        
        # Use the cached query method
        content = self.query_with_cache(prompt)
        
        if content:
            # Parse the response to extract structured information
            parsed_result = self.parse_fact_check_response(content, claim, retrieved_articles)
            
            # Skip the secondary verification to save an API call
            # Directly calculate confidence based on existing information
            verified_result = self.calibrate_confidence(
                parsed_result, 
                claim, 
                evidence, 
                source_credibility, 
                entities
            )
            
            # Cache the result
            result_cache[cache_key] = verified_result
            return verified_result
        else:
            # If API fails, fall back to using alignment analysis for a basic response
            return self._generate_fallback_response(claim, evidence, retrieved_articles, entities)
     
     
    def verify_fact_check_result(self, result: Dict[str, Any], claim: str, evidence: str) -> Dict[str, Any]:
        """Verify the fact-check result using a second pass of analysis
        
        Args:
            result: The initial fact-check result
            claim: The original claim
            evidence: The evidence text
            
        Returns:
            Verified and potentially adjusted result
        """
        # Extract the verdict and reasoning
        verdict = result.get("verdict", "Unverified")
        confidence = result.get("confidence", 50)
        reasoning = result.get("reasoning", "")
        
        # Prepare the verification prompt
        verification_prompt = f"""
    Review the following fact-check analysis for accuracy and bias:

    Original claim: "{claim}"

    Initial verdict: {verdict}
    Initial confidence: {confidence}%
    Reasoning: {reasoning}

    Please critically evaluate this analysis by checking for:
    1. Is the verdict consistent with the reasoning provided?
    2. Is the confidence level appropriate given the evidence strength?
    3. Are there any logical errors or biases in the reasoning?
    4. Is there a more appropriate verdict based on standard fact-checking principles?

    Provide your assessment:
    1. Should the verdict be changed? If so, to what?
    2. Should the confidence level be adjusted? If so, to what percentage?
    3. What are the strengths and weaknesses of the original analysis?
    """
        
        try:
            # Make second API request to Spark LLM
            payload = {
                "model": "4.0Ultra",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a critical fact-checking reviewer who evaluates the quality and accuracy of fact-check analyses."
                    },
                    {
                        "role": "user",
                        "content": verification_prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 1000,
                "stream": False
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_password}"
            }
            
            verification_response = requests.post(self.api_url, headers=headers, json=payload)
            verification_response.raise_for_status()
            verification_result = verification_response.json()
            
            if verification_result.get("code") == 0:
                verification_content = verification_result["choices"][0]["message"]["content"]
                
                # Extract any suggested verdict changes
                change_verdict_match = re.search(r"verdict be changed\?.*?(True|False|Partially True|Unverified)", verification_content, re.DOTALL | re.IGNORECASE)
                if change_verdict_match and "yes" in verification_content.lower():
                    new_verdict = change_verdict_match.group(1)
                    result["verdict"] = new_verdict
                    result["verification_note"] = "Verdict was adjusted during verification"
                
                # Extract any suggested confidence changes
                confidence_match = re.search(r"confidence level be adjusted\?.*?(\d+)%", verification_content, re.DOTALL | re.IGNORECASE)
                if confidence_match and "yes" in verification_content.lower():
                    new_confidence = int(confidence_match.group(1))
                    result["confidence"] = new_confidence
                    if "verification_note" in result:
                        result["verification_note"] += " with adjusted confidence"
                    else:
                        result["verification_note"] = "Confidence was adjusted during verification"
                
                # Add verification analysis to result
                result["verification_analysis"] = verification_content
            
            return result
            
        except Exception as e:
            logger.error(f"Error during verification: {e}")
            # If verification fails, return the original result
            return result
     
    def parse_fact_check_response(self, content: str, claim: str, retrieved_articles: List[Dict[str, str]]) -> Dict[str, Any]:
        """Optimized parser that handles incomplete or malformed responses better
        
        Args:
            content: The response content from Spark LLM
            claim: The original claim
            retrieved_articles: The articles used for fact-checking
            
        Returns:
            Structured fact-check result
        """
        # Initialize with default values to handle parsing errors gracefully
        result = {
            "claim": claim,
            "full_response": content,
            "retrieved_article_count": len(retrieved_articles),
            "verdict": "Unverified",
            "confidence": 50,
            "explanation": "Unable to determine the veracity of this claim."
        }
        
        # Extract verdict with improved regex
        verdict_match = re.search(r"Verdict:\s*(True|Partially True|False|Unverified)", content, re.IGNORECASE)
        if verdict_match:
            result["verdict"] = verdict_match.group(1)
        
        # Extract confidence
        confidence_match = re.search(r"Confidence:\s*(\d+)%?", content)
        if confidence_match:
            result["confidence"] = int(confidence_match.group(1))
        
        # Extract explanation
        explanation_match = re.search(r"Explanation:\s*(.*?)(?:\n\n|$)", content, re.DOTALL)
        if explanation_match:
            result["explanation"] = explanation_match.group(1).strip()
        
        # Add article sources
        if retrieved_articles:
            result["sources"] = [article.get("source", "Unknown source") for article in retrieved_articles[:2]]
        
        return result
     
     
    
    def _generate_fallback_response(self, claim: str, evidence: str, retrieved_articles: List[Dict[str, str]], entities: Dict[str, List[str]] = None) -> Dict[str, Any]:
        """Generate a fallback response when the API fails
        
        Args:
            claim: The claim to analyze
            evidence: The evidence text
            retrieved_articles: The retrieved articles
            entities: Optional entities dictionary
            
        Returns:
            A fallback fact-check response
        """
        # Analyze claim-evidence alignment for basic assessment
        alignment = self.analyze_claim_evidence_alignment(claim, evidence)
        similarity = alignment["semantic_similarity"]
        confidence = int(alignment["confidence"] * 100)
        
        # Get entity overlap information
        entity_overlap = alignment["entity_overlap"]["entity_coverage"] if entities else 0.5
        
        # Determine fallback verdict based on similarity and entity overlap
        if similarity > 0.8 and entity_overlap > 0.7:
            verdict = "True"
            explanation = "The claim appears to be supported by the evidence, with strong semantic similarity and entity matches."
        elif similarity > 0.6 or entity_overlap > 0.6:
            verdict = "Partially True"
            explanation = "The claim has some support in the evidence but may contain unverified elements."
        elif similarity < 0.3 and entity_overlap < 0.3:
            verdict = "False"
            explanation = "The claim contradicts available evidence or lacks sufficient support in reliable sources."
        else:
            verdict = "Unverified"
            explanation = "There is insufficient evidence to verify this claim."
        
        # Create a structured response
        fallback_response = {
            "verdict": verdict,
            "confidence": confidence,
            "reasoning": f"""
    Fallback analysis (API unavailable):
    - Semantic similarity between claim and evidence: {similarity:.2f}
    - Entity overlap: {entity_overlap:.2f}
    - Evidence sources: {len(retrieved_articles)} articles retrieved
            """,
            "explanation": explanation,
            "sources": [article.get("source", "Unknown Source") for article in retrieved_articles[:3]],
            "is_fallback": True
        }
        
        return fallback_response
     
     
    
    def calibrate_confidence(self, result: Dict[str, Any], claim: str, evidence: str, source_credibility: float = None, entities: Dict[str, List[str]] = None) -> Dict[str, Any]:
        """Calibrate the confidence score based on multiple factors
        
        Args:
            result: The initial fact-check result
            claim: The original claim
            evidence: The evidence text
            source_credibility: Optional source credibility score
            entities: Optional entities dictionary
            
        Returns:
            Result with calibrated confidence
        """
        initial_confidence = result.get("confidence", 50)
        
        # If this is a fallback result, use as is
        if result.get("is_fallback", False):
            return result
        
        # Calculate evidence strength factor
        evidence_length = min(len(evidence.split()) / 500, 1.0)  # Normalize evidence length
        evidence_strength = evidence_length * 0.5  # Longer evidence can contribute up to 0.5
        
        # Calculate semantic similarity factor
        alignment = self.analyze_claim_evidence_alignment(claim, evidence)
        semantic_similarity = alignment["semantic_similarity"]
        
        # Calculate entity verification factor
        entity_overlap = 0.5  # Default value
        if entities:
            entity_overlap = alignment["entity_overlap"]["entity_coverage"]
        
        # Source credibility factor
        credibility_factor = 0.5  # Default value
        if source_credibility is not None:
            credibility_factor = source_credibility
        
        # Verdict type factor - different verdicts have different baseline confidence
        verdict_type_factor = 0.0
        if result["verdict"] == "True":
            verdict_type_factor = 0.1
        elif result["verdict"] == "False":
            verdict_type_factor = 0.2  # False claims often have stronger evidence
        elif result["verdict"] == "Partially True":
            verdict_type_factor = -0.1  # Partial truths inherently have more uncertainty
        
        # Calculate adjustment based on all factors
        confidence_adjustment = (
            (semantic_similarity - 0.5) * 10 +  # Semantic similarity contribution
            (entity_overlap - 0.5) * 5 +        # Entity overlap contribution
            (evidence_strength - 0.5) * 5 +     # Evidence strength contribution
            (credibility_factor - 0.5) * 10 +   # Source credibility contribution
            verdict_type_factor * 10            # Verdict type contribution
        )
        
        # Apply the adjustment to the initial confidence
        calibrated_confidence = initial_confidence + confidence_adjustment
        
        # Ensure the confidence stays within reasonable bounds
        calibrated_confidence = max(min(calibrated_confidence, 98), 20)
        
        # Round to a natural-looking number
        calibrated_confidence = round(calibrated_confidence / 5) * 5
        
        # Update the result with calibrated confidence
        result["original_confidence"] = initial_confidence
        result["confidence"] = int(calibrated_confidence)
        result["confidence_factors"] = {
            "semantic_similarity": semantic_similarity,
            "entity_overlap": entity_overlap,
            "evidence_strength": evidence_strength,
            "source_credibility": credibility_factor,
            "verdict_type_factor": verdict_type_factor
        }
        
        return result
     
            
    def test_api_connection(self) -> bool:
        """Test the API connection with a simple, safe prompt
        
        Returns:
            True if connection successful, False otherwise
        """
        payload = {
            "model": "4.0Ultra",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, can you tell me about the weather today?"
                }
            ],
            "max_tokens": 50,
            "stream": False
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_password}"
        }
        
        try:
            logger.info("Testing Spark API connection...")
            response = requests.post(self.api_url, headers=headers, json=payload)
            logger.info(f"Test Response Status: {response.status_code}")
            
            result = response.json()
            if result.get("code") == 0:
                logger.info("✓ Spark API connection successful")
                return True
            else:
                logger.warning(f"× Spark API test failed: {result.get('message', 'Unknown error')}")
                logger.warning(f"Error code: {result.get('code', 'Unknown')}")
                return False
                
        except Exception as e:
            logger.error(f"Spark API test error: {str(e)}")
            return False
        
    
    def detect_title_content_contradiction(self, title: str, content: str) -> Dict[str, Any]:
        """
        Detect potential contradictions between title and content
        
        Args:
            title: Article title
            content: Article content
        
        Returns:
            Dictionary with contradiction analysis details
        """
        # Truncate content to manageable length
        content = content[:2000]  # Limit to prevent excessive API calls
        
        prompt = f"""
        Analyze the following news article for potential contradictions or misleading implications:
        
        Title: "{title}"
        Content: "{content}"
        
        Please evaluate the following aspects:
        1. Do the title and content match in meaning?
        2. Is the title potentially misleading compared to the actual content?
        3. Are there significant discrepancies between the title's implications and the article's substance?
        
        Response format:
        - Contradiction Severity: [Float between 0.0 and 1.0]
        0.0 = No contradiction
        1.0 = Extreme contradiction
        
        - Contradiction Type: 
        [Select from: "No Contradiction", "Slight Mismatch", "Moderate Misleading", "Significant Contradiction"]
        
        - Explanation: [Detailed reasoning for the contradiction assessment]
        """
        
        try:
            # Use Spark LLM to analyze contradiction (similar to fact_check method)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_password}"
            }
            
            payload = {
                "model": "4.0Ultra",
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are an expert fact-checker analyzing potential contradictions in news articles."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 500
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            # Parse the response
            content = result['choices'][0]['message']['content']
            
            # Extract key information using regex or parsing
            contradiction_severity_match = re.search(r'Contradiction Severity: (\d+\.\d+)', content)
            contradiction_type_match = re.search(r'Contradiction Type: (.*?)(\n|$)', content)
            explanation_match = re.search(r'Explanation: (.*)', content, re.DOTALL)
            
            contradiction_severity = float(contradiction_severity_match.group(1)) if contradiction_severity_match else 0.0
            contradiction_type = contradiction_type_match.group(1).strip() if contradiction_type_match else "Unknown"
            explanation = explanation_match.group(1).strip() if explanation_match else "No detailed explanation available"
            
            return {
                "severity": contradiction_severity,
                "type": contradiction_type,
                "explanation": explanation,
                "is_misleading": contradiction_severity > 0.5
            }
        
        except Exception as e:
            logger.error(f"Error detecting title-content contradiction: {e}")
            return {
                "severity": 0.0,
                "type": "Analysis Failed",
                "explanation": f"Could not analyze contradiction: {str(e)}",
                "is_misleading": False
            }

# Sentiment analysis component
class SentimentAnalyzer:
    """Component for analyzing sentiment and emotional manipulation in text"""
    
    def __init__(self):
        """Initialize the sentiment analyzer"""
        self.vader = SentimentIntensityAnalyzer()
        
        # Emotion keywords
        self.emotion_keywords = {
            'fear': ['fear', 'danger', 'threat', 'alarming', 'frightening', 'terrifying', 'scary', 'panic'],
            'anger': ['anger', 'rage', 'fury', 'outrage', 'angry', 'furious', 'enraged', 'mad'],
            'disgust': ['disgust', 'disgusting', 'repulsive', 'sickening', 'revolting', 'gross', 'vile'],
            'surprise': ['surprise', 'shocking', 'unbelievable', 'incredible', 'astonishing', 'stunning', 'dramatic'],
            'urgency': ['urgent', 'emergency', 'immediately', 'hurry', 'act now', 'don\'t wait', 'limited time'],
            'conspiracy': ['conspiracy', 'secret', 'cover-up', 'hidden', 'they don\'t want you to know', 'censored'],
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment in text using VADER
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of sentiment scores
        """
        return self.vader.polarity_scores(text)
    
    def analyze_emotion_keywords(self, text: str) -> Dict[str, float]:
        """Analyze the presence of emotion-laden keywords in the text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with emotion keyword scores
        """
        text_lower = text.lower()
        results = {}
        
        total_words = len(text_lower.split())
        if total_words == 0:
            return {emotion: 0.0 for emotion in self.emotion_keywords}
        
        for emotion, keywords in self.emotion_keywords.items():
            count = sum(text_lower.count(keyword) for keyword in keywords)
            # Normalize by text length to get a relative score
            score = count / total_words
            results[emotion] = float(score)
            
        return results
    
    def analyze_sentence_polarities(self, text: str) -> Dict[str, float]:
        """Analyze the distribution of sentence polarities in the text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with polarity statistics
        """
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        if not sentences:
            return {
                "polarity_mean": 0.0,
                "polarity_std": 0.0,
                "polarity_max": 0.0,
                "polarity_min": 0.0,
                "subjectivity_mean": 0.0,
                "extreme_sentence_ratio": 0.0,
            }
        
        # Calculate polarity for each sentence
        polarities = []
        extreme_sentences = 0
        
        for sentence in sentences:
            scores = self.analyze_sentiment(sentence)
            polarities.append(scores["compound"])
            
            # Count extreme polarity sentences (strong positive or negative)
            if abs(scores["compound"]) > 0.5:
                extreme_sentences += 1
        
        # Calculate statistics
        polarities_array = np.array(polarities)
        
        return {
            "polarity_mean": float(np.mean(polarities_array)),
            "polarity_std": float(np.std(polarities_array)),
            "polarity_max": float(np.max(polarities_array)),
            "polarity_min": float(np.min(polarities_array)),
            "extreme_sentence_ratio": float(extreme_sentences / len(sentences)),
        }
    
    def detect_emotional_manipulation(self, text: str) -> Dict[str, Any]:
        """Detect potential emotional manipulation in text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with manipulation metrics
        """
        # Get basic sentiment
        sentiment_scores = self.analyze_sentiment(text)
        
        # Get emotion keywords
        emotion_keywords = self.analyze_emotion_keywords(text)
        
        # Get sentence polarity statistics
        polarity_stats = self.analyze_sentence_polarities(text)
        
        # Calculate manipulation score based on several factors:
        # 1. High emotional content (extreme sentiment)
        # 2. High variance in sentence polarities (emotional rollercoaster)
        # 3. High frequency of emotional keywords
        # 4. High subjectivity
        
        factors = []
        
        # Factor 1: Extreme sentiment (very negative or very positive)
        sentiment_extremity = abs(sentiment_scores["compound"])
        factors.append(sentiment_extremity)
        
        # Factor 2: Sentence polarity variance
        polarity_variance = polarity_stats["polarity_std"]
        factors.append(min(polarity_variance * 2, 1.0))  # Scale up but cap at 1.0
        
        # Factor 3: Emotional keyword density
        emotion_keyword_score = sum(emotion_keywords.values())
        factors.append(min(emotion_keyword_score * 5, 1.0))  # Scale up but cap at 1.0
        
        # Factor 4: Extreme sentence ratio
        extreme_sentence_ratio = polarity_stats["extreme_sentence_ratio"]
        factors.append(extreme_sentence_ratio)
        
        # Calculate the overall manipulation score (weighted average)
        weights = [0.3, 0.2, 0.3, 0.2]  # Weights for each factor
        manipulation_score = sum(f * w for f, w in zip(factors, weights))
        
        # Qualitative assessment
        if manipulation_score < 0.3:
            manipulation_level = "LOW"
            explanation = "The text appears to be relatively neutral and factual, with limited emotional manipulation."
        elif manipulation_score < 0.6:
            manipulation_level = "MODERATE"
            explanation = "The text contains some emotional language and manipulation techniques, but is not extremely manipulative."
        else:
            manipulation_level = "HIGH"
            explanation = "The text shows strong signs of emotional manipulation, using extreme language and emotional appeals."
        
        # Add detailed explanation based on the highest contributing factors
        explanation_details = []
        if sentiment_extremity > 0.7:
            explanation_details.append("Uses extremely emotional language.")
        if polarity_variance > 0.5:
            explanation_details.append("Contains dramatic shifts in emotional tone.")
        if emotion_keyword_score > 0.2:
            top_emotions = sorted(emotion_keywords.items(), key=lambda x: x[1], reverse=True)[:2]
            emotion_str = " and ".join([f"{emotion}" for emotion, score in top_emotions if score > 0])
            if emotion_str:
                explanation_details.append(f"Appeals heavily to {emotion_str}.")
        if extreme_sentence_ratio > 0.5:
            explanation_details.append("Contains many emotionally charged statements.")
            
        if explanation_details:
            explanation += " " + " ".join(explanation_details)
        
         # Add propaganda technique detection
        propaganda_analysis = self.detect_propaganda_techniques(text)
        
        # Incorporate propaganda into manipulation score
        if propaganda_analysis["propaganda_score"] > 0:
            manipulation_score = 0.7 * manipulation_score + 0.3 * propaganda_analysis["propaganda_score"]
        
        # Update explanation
        if propaganda_analysis["has_propaganda"]:
            primary_technique = propaganda_analysis["primary_technique"]
            readable_technique = primary_technique.replace("_", " ").title()
            explanation += f" Uses '{readable_technique}' propaganda technique."
            
        # Determine dominant emotion
        if sentiment_scores["compound"] >= 0.5:
            dominant_emotion = "strongly positive"
        elif sentiment_scores["compound"] > 0 and sentiment_scores["compound"] < 0.5:
            dominant_emotion = "mildly positive"
        elif sentiment_scores["compound"] > -0.5 and sentiment_scores["compound"] <= 0:
            dominant_emotion = "mildly negative"
        else:
            dominant_emotion = "strongly negative"
        
        # Add dominant emotion from keywords if available
        keyword_emotions = [(emotion, score) for emotion, score in emotion_keywords.items() if score > 0.05]
        if keyword_emotions:
            top_keyword_emotion = max(keyword_emotions, key=lambda x: x[1])[0]
            dominant_emotion += f" with {top_keyword_emotion} undertones"
        
        return {
            "sentiment_scores": sentiment_scores,
            "emotional_intensity": sentiment_extremity,
            "is_emotionally_charged": manipulation_score > 0.5,
            "dominant_emotion": dominant_emotion,
            "manipulation_score": float(manipulation_score),
            "manipulation_level": manipulation_level,
            "explanation": explanation,
            "details": {
                "emotion_keywords": emotion_keywords,
                "polarity_stats": polarity_stats,
                "contributing_factors": {
                    "sentiment_extremity": float(sentiment_extremity),
                    "polarity_variance": float(polarity_variance),
                    "emotion_keyword_density": float(emotion_keyword_score),
                    "extreme_sentence_ratio": float(extreme_sentence_ratio)
                },
                "propaganda_analysis": propaganda_analysis
            }
        }
        
    def detect_propaganda_techniques(self, text: str) -> Dict[str, Any]:
        """
        Detect propaganda techniques used in text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with detected propaganda techniques
        """
        # Define propaganda technique patterns and keywords
        propaganda_techniques = {
            "name_calling": [
                "thug", "radical", "terrorist", "commie", "fascist", "libtard", 
                "nutjob", "traitor", "crook", "criminal", "extremist"
            ],
            "bandwagon": [
                "everyone knows", "majority believes", "people agree", 
                "most people", "widely accepted", "general consensus"
            ],
            "fear_mongering": [
                "catastrophe", "crisis", "threat", "danger", "disaster", "terrifying",
                "devastating", "alarming", "frightening", "panic"
            ],
            "appeal_to_authority": [
                "experts say", "scientists confirm", "doctors recommend", 
                "studies show", "research confirms", "according to authorities"
            ],
            "false_dilemma": [
                "either", "or", "there are only two", "it's either", "it's one or the other",
                "have no choice", "must choose between"
            ],
            "strawman": [
                "claim", "argue", "believe", "want", "they say", "they think", 
                "they believe", "they want", "position"
            ],
            "ad_hominem": [
                "corrupt", "dishonest", "liar", "incompetent", "stupid", 
                "evil", "greedy", "selfish"
            ],
            "whataboutism": [
                "what about", "but what about", "compared to", "instead of focusing on"
            ],
        }
        
        text_lower = text.lower()
        detected = {}
        
        # Check for each technique
        for technique, keywords in propaganda_techniques.items():
            hits = []
            for keyword in keywords:
                if keyword in text_lower:
                    hits.append(keyword)
            
            if hits:
                detected[technique] = {
                    "detected": True,
                    "match_count": len(hits),
                    "examples": hits[:5]  # Limit to 5 examples
                }
        
        # Calculate overall propaganda score
        technique_count = len(detected)
        propaganda_score = min(technique_count / len(propaganda_techniques), 1.0)
        
        # Determine primary technique
        primary_technique = None
        max_hits = 0
        
        for technique, data in detected.items():
            if data["match_count"] > max_hits:
                max_hits = data["match_count"]
                primary_technique = technique
        
        return {
            "detected_techniques": detected,
            "technique_count": technique_count,
            "propaganda_score": propaganda_score,
            "primary_technique": primary_technique,
            "has_propaganda": technique_count > 0
        }


# Source credibility analysis component
class CredibilityAnalyzer:
    """Component for analyzing the credibility of news sources and authors"""
    
    def __init__(self):
        """Initialize the credibility analyzer"""
        # Load or define known credible sources (in a real system, this would be a comprehensive database)
        self.credible_news_domains = {
            'reuters.com': 0.95,
            'apnews.com': 0.95,
            'bbc.com': 0.92,
            'bbc.co.uk': 0.92,
            'nytimes.com': 0.90,
            'wsj.com': 0.90,
            'washingtonpost.com': 0.88,
            'economist.com': 0.90,
            'npr.org': 0.88,
            'theguardian.com': 0.85,
            'bloomberg.com': 0.87,
            'cnn.com': 0.80,
            'nbcnews.com': 0.80,
            'abcnews.go.com': 0.80,
            'cbsnews.com': 0.80,
            'politico.com': 0.82,
            'thehill.com': 0.80,
            'usatoday.com': 0.78,
            'latimes.com': 0.82,
            'time.com': 0.85,
            'newsweek.com': 0.75,
            'theatlantic.com': 0.85,
            'newyorker.com': 0.87,
            'huffpost.com': 0.70,
            'vox.com': 0.75,
            'slate.com': 0.72,
            'foxnews.com': 0.70,
            'msnbc.com': 0.70,
        }
    
    def analyze_source_credibility(self, domain: str) -> Dict[str, Any]:
        """Analyze the credibility of a news source by its domain
        
        Args:
            domain: Domain name of the source
            
        Returns:
            Dictionary with credibility metrics
        """
        domain = domain.lower()  # Normalize domain
        
        # Check if domain is in our database of known sources
        base_score = self.credible_news_domains.get(domain)
        
        # If not in database, we do some heuristic analysis
        if base_score is None:
            # Check for educational or government domains
            if domain.endswith('.edu'):
                base_score = 0.85
            elif domain.endswith('.gov'):
                base_score = 0.90
            elif domain.endswith('.org'):
                base_score = 0.75  # .org sites can vary widely in credibility
            else:
                # Default score for unknown domains
                base_score = 0.50
        
        credibility_level = "UNKNOWN"
        explanation = ""
        
        if base_score >= 0.85:
            credibility_level = "HIGH"
            explanation = f"{domain} is recognized as a highly credible news source with established fact-checking processes."
        elif base_score >= 0.70:
            credibility_level = "MEDIUM"
            explanation = f"{domain} is a generally reliable source but may have some bias in reporting."
        elif base_score >= 0.50:
            credibility_level = "LOW_MEDIUM"
            explanation = f"{domain} has moderate credibility but may have significant bias or occasional accuracy issues."
        else:
            credibility_level = "LOW"
            explanation = f"{domain} has limited established credibility or is not a recognized mainstream news source."
            
        return {
            "domain": domain,
            "credibility_score": float(base_score),
            "credibility_level": credibility_level,
            "explanation": explanation
        }
    
    def analyze_author_credibility(self, author: str) -> Dict[str, Any]:
        """Analyze the credibility of an author (basic implementation)
        
        Args:
            author: Author name
            
        Returns:
            Dictionary with author credibility assessment
        """
        # In a real system, this would check against a database of known authors
        # Here we're implementing a basic version that looks for "staff" or unknown authors
        
        if not author or author.lower() in ["unknown", "unknown author", ""]:
            return {
                "author": author,
                "credibility_factor": 0.0,
                "explanation": "Unknown author reduces credibility. Articles without clear attribution are less verifiable."
            }
        
        if any(term in author.lower() for term in ["staff", "editor", "reporter", "correspondent"]):
            return {
                "author": author,
                "credibility_factor": 0.6,
                "explanation": "Generic staff attribution. While from an organization, specific author would increase credibility."
            }
        
        # Basic check for names that look legitimate (first and last name)
        name_parts = [part for part in author.split() if len(part) > 1 and part[0].isupper()]
        if len(name_parts) >= 2:
            return {
                "author": author,
                "credibility_factor": 0.8,
                "explanation": "Named author increases credibility as it provides accountability and verification possibilities."
            }
        
        return {
            "author": author,
            "credibility_factor": 0.5,
            "explanation": "Author information provides some accountability, but may need verification."
        }
        
    def analyze_date_credibility(self, date_str: str, claim_text: str) -> Dict[str, Any]:
        """Analyze the credibility impact of the publication date
        
        Args:
            date_str: Publication date string
            claim_text: Text of the claim being analyzed
            
        Returns:
            Dictionary with date credibility assessment
        """
        if not date_str or date_str.lower() in ["unknown", "unknown date", ""]:
            return {
                "date": "Unknown",
                "credibility_factor": 0.0,
                "recency": "Unknown",
                "explanation": "Missing publication date reduces credibility. Cannot verify timeliness or context."
            }
        
        try:
            # Try to parse date - handle multiple formats
            from datetime import datetime
            import re
            
            # Clean up the date string
            date_str = date_str.strip()
            
            # Handle "Updated" prefix
            if date_str.startswith("Updated"):
                date_str = date_str.replace("Updated", "").strip()
            
            # Try different date parsing approaches
            try:
                # First try: direct parsing
                pub_date = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                try:
                    # Second try: parse with dateutil
                    import dateutil.parser
                    pub_date = dateutil.parser.parse(date_str)
                except:
                    # Third try: extract date using regex
                    date_pattern = r'(\d{1,2}:\d{2}\s*[AP]M\s*[A-Z]{3,4},\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s*\d{4})'
                    match = re.search(date_pattern, date_str)
                    if match:
                        date_str = match.group(1)
                        pub_date = dateutil.parser.parse(date_str)
                    else:
                        raise ValueError(f"Could not parse date: {date_str}")
            
            # Ensure both dates are timezone-naive for comparison
            if pub_date.tzinfo is not None:
                pub_date = pub_date.replace(tzinfo=None)
            
            current_date = datetime.now()
            
            # Calculate days difference
            days_diff = (current_date - pub_date).days
            
            # Extract years from the claim (important for historical claims)
            year_pattern = r'\b(19\d{2}|20\d{2})\b'
            years_in_claim = re.findall(year_pattern, claim_text)
            
            # If claim mentions specific years and article is from that time, it may be more credible
            if years_in_claim:
                years_mentioned = [int(year) for year in years_in_claim]
                pub_year = pub_date.year
                
                if pub_year in years_mentioned or any(abs(pub_year - year) <= 1 for year in years_mentioned):
                    return {
                        "date": pub_date.strftime("%Y-%m-%d"),
                        "days_ago": days_diff,
                        "credibility_factor": 0.9,
                        "recency": "Contemporaneous with events",
                        "explanation": f"Publication date ({pub_date.strftime('%Y-%m-%d')}) is contemporaneous with events mentioned in the claim, increasing credibility."
                    }
            
            # Assess recency
            if days_diff < 7:
                recency = "Very recent"
                factor = 0.9  # Very recent news may lack full fact-checking
            elif days_diff < 30:
                recency = "Recent"
                factor = 0.95  # Recent but enough time for fact-checking
            elif days_diff < 365:
                recency = "Within past year"
                factor = 0.85  # Relevant but may miss recent developments
            elif days_diff < 365 * 2:
                recency = "1-2 years old"
                factor = 0.7
            else:
                recency = "Older than 2 years"
                factor = 0.5  # May be outdated for current events
                
                # If claim is about current events but article is old, reduce more
                current_event_indicators = ["today", "yesterday", "this week", "this month", "this year", "currently", "now"]
                if any(indicator in claim_text.lower() for indicator in current_event_indicators):
                    factor = 0.3
                    recency = "Outdated for current claim"
            
            return {
                "date": pub_date.strftime("%Y-%m-%d"),
                "days_ago": days_diff,
                "credibility_factor": factor,
                "recency": recency,
                "explanation": f"Publication date: {pub_date.strftime('%Y-%m-%d')} ({recency}). {days_diff} days ago."
            }
            
        except Exception as e:
            logger.error(f"Error parsing date: {e}")
            return {
                "date": date_str,
                "credibility_factor": 0.3,
                "recency": "Unknown format",
                "explanation": f"Unparseable publication date '{date_str}' reduces credibility. Cannot verify timeliness."
            }
    
    def assess_overall_source_credibility(self, domain: str, author: str, date_str: str, claim_text: str) -> Dict[str, Any]:
        """Assess the overall credibility of a source based on multiple factors
        
        Args:
            domain: Source domain
            author: Author name
            date_str: Publication date string
            claim_text: Text of the claim
            
        Returns:
            Dictionary with overall credibility assessment
        """
        # Get individual credibility factors
        source_cred = self.analyze_source_credibility(domain)
        author_cred = self.analyze_author_credibility(author)
        date_cred = self.analyze_date_credibility(date_str, claim_text)
        
        # Calculate weighted overall score
        weights = {
            "source": 0.5,  # Source/domain is most important
            "author": 0.3,  # Author attribution is next
            "date": 0.2     # Date is still important but less weighted
        }
        
        overall_score = (
            source_cred["credibility_score"] * weights["source"] +
            author_cred["credibility_factor"] * weights["author"] +
            date_cred["credibility_factor"] * weights["date"]
        )
        
        # Determine overall credibility level
        if overall_score >= 0.85:
            level = "HIGH"
            explanation = "This is a highly credible source. The information likely underwent editorial review and fact-checking."
        elif overall_score >= 0.70:
            level = "MEDIUM_HIGH"
            explanation = "This is a generally credible source with some minor concerns."
        elif overall_score >= 0.50:
            level = "MEDIUM"
            explanation = "This source has moderate credibility. Verify with additional sources if possible."
        elif overall_score >= 0.30:
            level = "LOW_MEDIUM"
            explanation = "This source has questionable credibility. Treat information with caution."
        else:
            level = "LOW"
            explanation = "This source has low credibility. Information should be verified with more reliable sources."
        
        # Build detailed explanation
        details = [
            f"Source: {source_cred['explanation']}",
            f"Author: {author_cred['explanation']}",
            f"Date: {date_cred['explanation']}"
        ]
        
        return {
            "overall_credibility_score": float(overall_score),
            "credibility_level": level,
            "explanation": explanation,
            "details": details,
            "factors": {
                "source": source_cred,
                "author": author_cred,
                "date": date_cred
            }
        }
    
    def calculate_trust_lens_score(self, source_credibility, factual_match, tone_neutrality, source_transparency):
        """
        Calculate a comprehensive trust score
        
        Weights can be adjusted based on empirical testing
        """
        weights = {
            'source_credibility': 0.4,
            'factual_match': 0.3,
            'tone_neutrality': 0.2,
            'source_transparency': 0.1
        }
        
        trust_score = (
            source_credibility * weights['source_credibility'] +
            factual_match * weights['factual_match'] +
            tone_neutrality * weights['tone_neutrality'] +
            source_transparency * weights['source_transparency']
        )
        
        return min(max(trust_score, 0), 1)  # Ensure score between 0-1


# Add these classes after the CredibilityAnalyzer class in Try_train.py

# Add this class after PromptQualityAnalyzer in Try_train.py
class AIContentDetector:
    """Enhanced component for detecting if content was likely generated by AI and categorizing news type"""
    
    def __init__(self, spark_api_password=None):
        """Initialize with Spark API credentials"""
        self.spark_api_password = spark_api_password or os.getenv('SPARK_API_PASSWORD')
        self.api_url = "https://spark-api-open.xf-yun.com/v1/chat/completions"
        
        # Linguistic patterns common in AI-generated text
        self.ai_patterns = [
            r'\b(moreover|furthermore|additionally|consequently)\b',
            r'\b(in conclusion|to summarize|to sum up)\b',
            r'\b(it is important to note that|it should be noted that)\b',
            r'\b(various|numerous|multiple|several|diverse)\b',
            r'\b(as mentioned earlier|as previously stated|as noted above)\b',
            # Add more AI-specific patterns
            r'\b(on the one hand|on the other hand)\b',
            r'\b(firstly|secondly|thirdly|lastly)\b',
            r'\b(in terms of|with respect to|with regard to)\b',
            r'\b(it can be argued that|it is worth noting that)\b'
        ]
        
        # Category features
        self.category_keywords = {
            'official_news': ['press release', 'official statement', 'announces', 'government', 'authority', 'ministry', 'department'],
            'educational': ['research', 'study', 'university', 'professor', 'academic', 'science', 'findings', 'education'],
            'opinion': ['opinion', 'editorial', 'commentary', 'perspective', 'viewpoint', 'column'],
            'entertainment': ['celebrity', 'movie', 'entertainment', 'film', 'music', 'TV', 'star', 'actress', 'actor'],
            'sports': ['game', 'match', 'player', 'team', 'tournament', 'championship', 'sports', 'league', 'score'],
            'technology': ['tech', 'technology', 'app', 'software', 'digital', 'internet', 'device', 'startup'],
            'health': ['health', 'medical', 'disease', 'treatment', 'patient', 'doctor', 'hospital', 'medicine']
        }
    
    def _check_basic_ai_patterns(self, text: str) -> Dict[str, float]:
        """Check for common AI writing patterns
        
        Args:
            text: The content to analyze
            
        Returns:
            Dictionary with pattern match statistics
        """
        # Count pattern occurrences
        pattern_counts = {}
        text_lower = text.lower()
        
        for pattern in self.ai_patterns:
            matches = re.findall(pattern, text_lower)
            pattern_counts[pattern] = len(matches)
        
        # Calculate overall score based on pattern density
        word_count = len(text_lower.split())
        if word_count == 0:
            return {"ai_pattern_score": 0.0, "patterns_found": pattern_counts}
        
        total_matches = sum(pattern_counts.values())
        pattern_density = total_matches / (word_count / 100)  # Matches per 100 words
        
        # Normalize to a 0-1 score
        ai_pattern_score = min(1.0, pattern_density / 5)  # Cap at 1.0
        
        return {
            "ai_pattern_score": ai_pattern_score,
            "patterns_found": pattern_counts,
            "pattern_density": pattern_density
        }
    
    def _identify_content_category(self, text: str) -> Dict[str, Any]:
        """Identify the most likely category of the content
        
        Args:
            text: The content to analyze
            
        Returns:
            Dictionary with category analysis
        """
        text_lower = text.lower()
        category_scores = {}
        
        # Calculate scores for each category based on keyword matches
        for category, keywords in self.category_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            category_scores[category] = matches / len(keywords)  # Normalize by keyword count
        
        # Find the top categories
        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        top_category = sorted_categories[0][0] if sorted_categories else "general"
        top_score = sorted_categories[0][1] if sorted_categories else 0
        
        # Check if the top category is significantly higher than others
        is_clear_category = len(sorted_categories) > 1 and sorted_categories[0][1] > sorted_categories[1][1] * 1.5
        
        return {
            "primary_category": top_category,
            "category_confidence": min(1.0, top_score * 2),  # Scale up but cap at 1.0
            "is_clear_category": is_clear_category,
            "category_scores": category_scores,
            "top_categories": sorted_categories[:3] if len(sorted_categories) >= 3 else sorted_categories
        }
    
    def _analyze_structural_features(self, text: str) -> Dict[str, Any]:
        """Analyze structural features that might indicate AI generation
        
        Args:
            text: The content to analyze
            
        Returns:
            Dictionary with structural analysis
        """
        # Split into paragraphs
        paragraphs = [p for p in text.split('\n') if p.strip()]
        
        if not paragraphs:
            return {
                "avg_paragraph_length": 0,
                "paragraph_length_variance": 0,
                "structure_score": 0
            }
        
        # Calculate paragraph lengths
        paragraph_lengths = [len(p.split()) for p in paragraphs]
        avg_paragraph_length = sum(paragraph_lengths) / len(paragraph_lengths)
        
        # Calculate variance in paragraph lengths
        if len(paragraph_lengths) > 1:
            variance = sum((length - avg_paragraph_length) ** 2 for length in paragraph_lengths) / len(paragraph_lengths)
            std_dev = variance ** 0.5
            normalized_std_dev = std_dev / avg_paragraph_length if avg_paragraph_length > 0 else 0
        else:
            normalized_std_dev = 0
        
        # AI-generated content often has very consistent paragraph lengths
        # Lower variance might indicate AI-generation
        structure_score = max(0, min(1, 1 - normalized_std_dev))
        
        # Count sentences per paragraph
        sentence_counts = []
        for paragraph in paragraphs:
            sentences = re.split(r'[.!?]+', paragraph)
            sentences = [s for s in sentences if s.strip()]
            sentence_counts.append(len(sentences))
        
        # Calculate sentence length variance
        if paragraphs:
            sentences = re.split(r'[.!?]+', ' '.join(paragraphs))
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if sentences:
                sentence_lengths = [len(s.split()) for s in sentences]
                avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)
                
                if len(sentence_lengths) > 1:
                    sentence_variance = sum((length - avg_sentence_length) ** 2 for length in sentence_lengths) / len(sentence_lengths)
                    sentence_std_dev = sentence_variance ** 0.5
                    sentence_std_ratio = sentence_std_dev / avg_sentence_length if avg_sentence_length > 0 else 0
                else:
                    sentence_std_ratio = 0
                
                # AI often has consistent sentence lengths
                sentence_structure_score = max(0, min(1, 1 - sentence_std_ratio))
                
                # Combine paragraph and sentence structure scores
                structure_score = (structure_score + sentence_structure_score) / 2
        
        return {
            "avg_paragraph_length": avg_paragraph_length,
            "paragraph_length_variance": normalized_std_dev,
            "structure_score": structure_score,
            "sentence_counts": sentence_counts
        }
    
    def detect_ai_content(self, text: str) -> Dict[str, Any]:
        """Detect if content was likely generated by AI and categorize it
        
        Args:
            text: The content to analyze
            
        Returns:
            Dictionary with AI detection results and content categorization
        """
        # Check cache first
        cache_key = f"ai_detection_{hash(text)}"
        if cache_key in result_cache:
            return result_cache[cache_key]
        
        # Run basic pattern analysis
        pattern_analysis = self._check_basic_ai_patterns(text)
        
        # Run structural analysis
        structure_analysis = self._analyze_structural_features(text)
        
        # Run category analysis
        category_analysis = self._identify_content_category(text)
        
        # Make preliminary AI detection based on patterns and structure
        preliminary_ai_score = 0.4 * pattern_analysis["ai_pattern_score"] + 0.6 * structure_analysis["structure_score"]
        
        try:
            # Use Spark LLM for more sophisticated detection
            meta_prompt = f"""
            Analyze the following text and determine if it was likely written by AI or by a human. 
            Also categorize the type of content.
            
            Text to analyze:
            "{text[:2000]}..." [truncated due to length]
            
            Please consider:
            1. Writing style (repetitive phrases, formulaic structure)
            2. Language naturality (awkward phrasing, unnatural transitions)
            3. Content depth and complexity
            4. Variation in sentence structure
            5. Presence of nuance and subtle contextual understanding
            
            Also determine what category the content belongs to:
            - Official News (government announcements, press releases)
            - Educational (research, academic articles, explanatory content)
            - Opinion/Editorial (commentary, personal viewpoints)
            - Entertainment (celebrity news, culture)
            - Sports News
            - Technology News
            - Health/Medical News
            - Other (specify)
            
            Format your response exactly as follows:
            AI_SCORE: [numerical score from 0.0 to 1.0, where 1.0 means definitely AI-generated]
            CATEGORY: [primary category]
            SUBCATEGORY: [more specific if possible]
            REASONING: [brief explanation of your assessment]
            """
            
            # Make API request to Spark LLM
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.spark_api_password}"
            }
            
            payload = {
                "model": "4.0Ultra",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert at detecting AI-generated content and categorizing text."
                    },
                    {
                        "role": "user",
                        "content": meta_prompt
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 500,
                "stream": False
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Parse Spark LLM response
            if result.get("code") == 0:
                content = result["choices"][0]["message"]["content"]
                
                # Extract AI score
                ai_score_match = re.search(r'AI_SCORE:\s*(\d+\.\d+)', content)
                ai_score = float(ai_score_match.group(1)) if ai_score_match else preliminary_ai_score
                
                # Extract category
                category_match = re.search(r'CATEGORY:\s*([^\n]+)', content)
                category = category_match.group(1).strip() if category_match else category_analysis["primary_category"]
                
                # Extract subcategory
                subcategory_match = re.search(r'SUBCATEGORY:\s*([^\n]+)', content)
                subcategory = subcategory_match.group(1).strip() if subcategory_match else "General"
                
                # Extract reasoning
                reasoning_match = re.search(r'REASONING:\s*(.*?)$', content, re.DOTALL)
                reasoning = reasoning_match.group(1).strip() if reasoning_match else "No detailed reasoning provided."
                
                # Combine LLM assessment with pattern analysis
                final_ai_score = 0.7 * ai_score + 0.3 * preliminary_ai_score
                
                # Determine AI generated verdict
                if final_ai_score >= 0.8:
                    ai_verdict = "Highly likely AI-generated"
                    emoji = "🤖"
                    confidence_text = "with high confidence"
                    is_ai = True
                elif final_ai_score >= 0.6:
                    ai_verdict = "Possibly AI-generated"
                    emoji = "🤖❓"
                    confidence_text = "with moderate confidence"
                    is_ai = True
                elif final_ai_score >= 0.4:
                    ai_verdict = "Uncertain origin"
                    emoji = "❓"
                    confidence_text = "with low confidence"
                    is_ai = None
                elif final_ai_score >= 0.2:
                    ai_verdict = "Likely human-written"
                    emoji = "👤❓"
                    confidence_text = "with moderate confidence"
                    is_ai = False
                else:
                    ai_verdict = "Very likely human-written"
                    emoji = "👤"
                    confidence_text = "with high confidence"
                    is_ai = False
                
                # Create friendly display messages
                if is_ai is True:
                    display_message = f"This content was likely written by AI {confidence_text}"
                elif is_ai is False:
                    display_message = f"This content was likely written by a human {confidence_text}"
                else:
                    display_message = f"The origin of this content is uncertain"
                
                # Prepare the result with enhanced user-friendly format
                result_data = {
                    "ai_score": final_ai_score,
                    "ai_verdict": ai_verdict,
                    "content_category": category,
                    "content_subcategory": subcategory,
                    "reasoning": reasoning,
                    "pattern_analysis": pattern_analysis,
                    "structure_analysis": structure_analysis,
                    "category_analysis": category_analysis,
                    # Enhanced display elements
                    "emoji": emoji,
                    "display_message": display_message,
                    "is_ai": is_ai,
                    "confidence_text": confidence_text,
                    "confidence_percentage": int(final_ai_score * 100),
                    # Linguistic traits
                    "linguistic_traits": self._extract_linguistic_traits(text, pattern_analysis)
                }
                
                # Cache the result
                result_cache[cache_key] = result_data
                return result_data
            else:
                # Fallback if API fails
                return self._fallback_detection(text, pattern_analysis, structure_analysis, category_analysis)
                
        except Exception as e:
            logger.error(f"Error in AI content detection: {e}")
            return self._fallback_detection(text, pattern_analysis, structure_analysis, category_analysis)
    
    def _extract_linguistic_traits(self, text: str, pattern_analysis: Dict) -> Dict[str, Any]:
        """Extract linguistic traits that could indicate AI or human authorship
        
        Args:
            text: The content to analyze
            pattern_analysis: Results of pattern analysis
            
        Returns:
            Dictionary with linguistic traits analysis
        """
        traits = {}
        
        # Check for personal pronouns (humans tend to use more)
        personal_pronouns = ['I', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves']
        pronoun_count = sum(len(re.findall(r'\b' + p + r'\b', text, re.IGNORECASE)) for p in personal_pronouns)
        word_count = len(text.split())
        
        # Calculate pronoun density
        pronoun_density = pronoun_count / word_count if word_count > 0 else 0
        traits["personal_pronoun_density"] = pronoun_density
        
        # Check for informal language
        informal_words = ['yeah', 'nah', 'cool', 'awesome', 'stuff', 'kinda', 'sorta', 'pretty much', 'you know']
        informal_count = sum(len(re.findall(r'\b' + w + r'\b', text.lower())) for w in informal_words)
        informal_density = informal_count / word_count if word_count > 0 else 0
        traits["informal_language_density"] = informal_density
        
        # Check for contractions (humans use more contractions)
        contractions = ["'s", "n't", "'m", "'re", "'ve", "'ll", "'d"]
        contraction_count = sum(text.count(c) for c in contractions)
        contraction_density = contraction_count / word_count if word_count > 0 else 0
        traits["contraction_density"] = contraction_density
        
        # Check sentence complexity
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        traits["avg_sentence_length"] = avg_sentence_length
        
        # Calculate a human-likeness score based on these traits
        human_score = (
            min(pronoun_density * 10, 1) * 0.3 +  # Personal pronouns contribute 30%
            min(informal_density * 20, 1) * 0.3 +  # Informal language contributes 30%
            min(contraction_density * 10, 1) * 0.4  # Contractions contribute 40%
        )
        
        traits["human_language_score"] = human_score
        
        return traits
    
    def _fallback_detection(self, text: str, pattern_analysis: Dict, structure_analysis: Dict, category_analysis: Dict) -> Dict[str, Any]:
        """Generate fallback AI detection when API is unavailable
        
        Args:
            text: The content to analyze
            pattern_analysis: Results of pattern analysis
            structure_analysis: Results of structure analysis
            category_analysis: Results of category analysis
            
        Returns:
            Dictionary with AI detection results using only rule-based methods
        """
        # Calculate AI score from pattern and structure analysis
        ai_score = 0.4 * pattern_analysis["ai_pattern_score"] + 0.6 * structure_analysis["structure_score"]
        
        # Extract linguistic traits even in fallback mode
        linguistic_traits = self._extract_linguistic_traits(text, pattern_analysis)
        
        # Adjust score based on linguistic traits
        human_language_score = linguistic_traits.get("human_language_score", 0)
        adjusted_ai_score = (ai_score * 0.7) - (human_language_score * 0.3)
        adjusted_ai_score = max(0, min(1, adjusted_ai_score))  # Keep between 0 and 1
        
        # Determine AI generated verdict
        if adjusted_ai_score >= 0.8:
            ai_verdict = "Highly likely AI-generated"
            emoji = "🤖"
            confidence_text = "with high confidence"
            is_ai = True
        elif adjusted_ai_score >= 0.6:
            ai_verdict = "Possibly AI-generated"
            emoji = "🤖❓"
            confidence_text = "with moderate confidence"
            is_ai = True
        elif adjusted_ai_score >= 0.4:
            ai_verdict = "Uncertain origin"
            emoji = "❓"
            confidence_text = "with low confidence"
            is_ai = None
        elif adjusted_ai_score >= 0.2:
            ai_verdict = "Likely human-written"
            emoji = "👤❓"
            confidence_text = "with moderate confidence"
            is_ai = False
        else:
            ai_verdict = "Very likely human-written"
            emoji = "👤"
            confidence_text = "with high confidence"
            is_ai = False
        
        # Create friendly display messages
        if is_ai is True:
            display_message = f"This content was likely written by AI {confidence_text}"
        elif is_ai is False:
            display_message = f"This content was likely written by a human {confidence_text}"
        else:
            display_message = f"The origin of this content is uncertain"
        
        # Basic reasoning based on patterns and structure
        reasoning_parts = []
        
        if pattern_analysis["ai_pattern_score"] > 0.6:
            reasoning_parts.append("Contains many linguistic patterns common in AI-generated text.")
        elif pattern_analysis["ai_pattern_score"] > 0.3:
            reasoning_parts.append("Contains some linguistic patterns that may suggest AI-generation.")
            
        if structure_analysis["structure_score"] > 0.7:
            reasoning_parts.append("Has very consistent paragraph and sentence structures typical of AI writing.")
        elif structure_analysis["structure_score"] > 0.4:
            reasoning_parts.append("Shows moderately consistent structural patterns.")
            
        if linguistic_traits["human_language_score"] > 0.7:
            reasoning_parts.append("Contains natural language patterns typical of human writing (contractions, personal pronouns, informal language).")
        elif linguistic_traits["human_language_score"] < 0.3:
            reasoning_parts.append("Lacks natural human writing elements like contractions and personal pronouns.")
            
        if not reasoning_parts:
            reasoning_parts.append("The text shows a mix of AI and human writing characteristics, making it difficult to definitively categorize.")
            
        reasoning = " ".join(reasoning_parts)
        
        return {
            "ai_score": adjusted_ai_score,
            "ai_verdict": ai_verdict,
            "content_category": category_analysis["primary_category"],
            "content_subcategory": "General",
            "reasoning": reasoning,
            "pattern_analysis": pattern_analysis,
            "structure_analysis": structure_analysis,
            "category_analysis": category_analysis,
            "note": "This is a fallback analysis due to API limitations.",
            # Enhanced display elements
            "emoji": emoji,
            "display_message": display_message,
            "is_ai": is_ai,
            "confidence_text": confidence_text,
            "confidence_percentage": int(adjusted_ai_score * 100),
            "linguistic_traits": linguistic_traits
        }

    def categorize_content(self, text: str) -> Dict[str, Any]:
        """Categorize the content type and detect AI generation likelihood
        
        Args:
            text: The content to analyze
            
        Returns:
            Dictionary with categorization results
        """
        # Check for AI patterns
        ai_patterns = self._check_basic_ai_patterns(text)
        
        # Determine content category
        category_scores = {}
        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword.lower() in text.lower())
            category_scores[category] = score
        
        # Get the dominant category
        dominant_category = max(category_scores.items(), key=lambda x: x[1])[0] if category_scores else "unknown"
        
        return {
            "ai_likelihood": ai_patterns.get("ai_score", 0.0),
            "category": dominant_category,
            "category_scores": category_scores,
            "pattern_matches": ai_patterns.get("matches", {})
        }


# Add this class to Try_train.py after the CredibilityAnalyzer class
class PromptQualityAnalyzer:
    """Component for analyzing the quality of user prompts and providing suggestions for improvement"""
    
    def __init__(self, spark_api_password=None):
        """Initialize with Spark API credentials"""
        self.spark_api_password = spark_api_password or os.getenv('SPARK_API_PASSWORD')
        self.api_url = "https://spark-api-open.xf-yun.com/v1/chat/completions"
        
    def analyze_prompt_quality(self, prompt: str) -> Dict[str, Any]:
        """Analyze the quality of a user prompt and provide suggestions
        
        Args:
            prompt: The user's prompt text
            
        Returns:
            Dictionary with quality assessment and suggestions
        """
        # Check cache first
        cache_key = f"prompt_quality_{hash(prompt)}"
        if cache_key in result_cache:
            return result_cache[cache_key]
        
        # Check if prompt is a URL
        is_url = re.match(r'^https?://', prompt.strip())
        
        # Check if prompt is too short
        too_short = len(prompt.split()) < 5 and not is_url
        
        # Check if prompt contains specific details
        has_specific_details = len(prompt.split()) > 10 and not is_url
        
        # For URLs, use different criteria
        if is_url:
            return {
                "is_good_prompt": True,
                "quality_score": 0.85,
                "is_url": True,
                "suggestions": ["URL detected. Ready to analyze this source."],
                "improved_prompt": prompt
            }
        
        # For very short prompts, provide immediate feedback without API call
        if too_short:
            return {
                "is_good_prompt": False,
                "quality_score": 0.3,
                "is_url": False,
                "suggestions": [
                    "Prompt is too brief. Add more details about the claim.",
                    "Include the source, date, or context of the information.",
                    "Specify what aspect of the claim you want to verify."
                ],
                "improved_prompt": prompt  # Keep original for very short prompts
            }
        
        # For moderately detailed prompts, use Spark LLM to analyze quality
        try:
            # Build the prompt
            meta_prompt = f"""
            Analyze the following user prompt for fact-checking. Evaluate the prompt quality and suggest improvements.
            
            User Prompt: "{prompt}"
            
            Analyze the prompt for:
            1. Specificity: How specific is the claim being made?
            2. Verifiability: Can the claim be verified with evidence?
            3. Context: Does the prompt provide enough context (e.g., date, source)?
            4. Clarity: Is the request clear about what needs to be verified?
            
            Rate the prompt quality on a scale from 0 to 1.
            
            Provide 2-3 specific suggestions to improve the prompt if needed.
            If the prompt is already high quality, acknowledge that.
            
            Format response as:
            QUALITY_SCORE: [0.0-1.0]
            SUGGESTIONS:
            - [Suggestion 1]
            - [Suggestion 2]
            IMPROVED_PROMPT: [A better version of their prompt]
            """
            
            # Make API request to Spark LLM
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.spark_api_password}"
            }
            
            payload = {
                "model": "4.0Ultra",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a prompt engineering expert who helps users create better prompts for fact-checking."
                    },
                    {
                        "role": "user",
                        "content": meta_prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 500,
                "stream": False
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Extract content
            if result.get("code") == 0:
                content = result["choices"][0]["message"]["content"]
                
                # Parse the structured response
                quality_score_match = re.search(r'QUALITY_SCORE:\s*(\d+\.\d+)', content)
                quality_score = float(quality_score_match.group(1)) if quality_score_match else 0.5
                
                # Extract suggestions (list items after SUGGESTIONS:)
                suggestions_section = re.search(r'SUGGESTIONS:(.*?)IMPROVED_PROMPT:', content, re.DOTALL)
                suggestions = []
                if suggestions_section:
                    suggestion_text = suggestions_section.group(1).strip()
                    for line in suggestion_text.split('\n'):
                        line = line.strip()
                        if line.startswith('-'):
                            suggestions.append(line[1:].strip())
                
                # Extract improved prompt
                improved_prompt_match = re.search(r'IMPROVED_PROMPT:(.*?)$', content, re.DOTALL)
                improved_prompt = improved_prompt_match.group(1).strip() if improved_prompt_match else prompt
                
                result_data = {
                    "is_good_prompt": quality_score >= 0.7,
                    "quality_score": quality_score,
                    "is_url": False,
                    "suggestions": suggestions if quality_score < 0.7 else ["Prompt is already well-structured for analysis."],
                    "improved_prompt": improved_prompt
                }
                
                # Cache the result
                result_cache[cache_key] = result_data
                return result_data
            else:
                # Fallback if API fails
                return self._generate_basic_quality_assessment(prompt)
                
        except Exception as e:
            logger.error(f"Error analyzing prompt quality: {e}")
            return self._generate_basic_quality_assessment(prompt)
    
    def _generate_basic_quality_assessment(self, prompt: str) -> Dict[str, Any]:
        """Generate a basic quality assessment without using the API
        
        Args:
            prompt: The user's prompt text
            
        Returns:
            Dictionary with quality assessment and suggestions
        """
        # Simple heuristic analysis
        words = prompt.split()
        word_count = len(words)
        
        # Calculate a basic quality score
        if word_count < 5:
            quality_score = 0.3
        elif word_count < 10:
            quality_score = 0.5
        elif word_count < 20:
            quality_score = 0.7
        else:
            quality_score = 0.8
        
        # Generate generic suggestions based on prompt length
        suggestions = []
        if word_count < 10:
            suggestions.append("Add more details about the claim you're verifying.")
            suggestions.append("Include information about where you encountered this claim.")
        if not any(entity in prompt.lower() for entity in ["who", "what", "when", "where", "why", "how"]):
            suggestions.append("Include specifics like who, what, when, where about the claim.")
        
        # If no suggestions were added, add a default one
        if not suggestions:
            suggestions.append("Consider adding more context about the source or timing of this information.")
        
        return {
            "is_good_prompt": quality_score >= 0.7,
            "quality_score": quality_score,
            "is_url": False,
            "suggestions": suggestions,
            "improved_prompt": prompt  # Keep original in basic assessment
        }

# Main Fake News Detection System
class FakeNewsDetectionSystem:
    """Complete system for fake news detection using Spark LLM and RAG"""
    
    def __init__(self):
        """Initialize the fake news detection system"""
        validate_credentials()
        
        self.search_engine = WebSearchEngine(GOOGLE_API_KEY, GOOGLE_CX)
        self.embedding_engine = EmbeddingEngine()
        self.faiss_indexer = FAISSIndexer()
        self.spark_llm = SparkLLMFactChecker(SPARK_API_PASSWORD)  # Replace Spark LLM with Spark LLM
        self.sentiment_analyzer = SentimentAnalyzer()
        self.entity_extractor = EntityExtractor()
        self.url_handler = URLHandler()
        self.credibility_analyzer = CredibilityAnalyzer()
        
        # Add the new components
        self.ai_detector = AIContentDetector(SPARK_API_PASSWORD)
        self.prompt_suggester = PromptQualityAnalyzer(SPARK_API_PASSWORD)

        # Test Spark API connection
        self.spark_llm.test_api_connection()

        # Initialize knowledge graph if credentials are available
        if NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD:
            self.knowledge_graph = KnowledgeGraph()
        else:
            self.knowledge_graph = None
            logger.warning("Knowledge Graph not initialized due to missing credentials")


    def analyze_claim(self, claim: str, use_rag: bool = True, use_kg: bool = True) -> Dict[str, Any]:
        """Analyze a claim for fake news detection with enhanced models
        
        Args:
            claim: The claim to analyze or a URL
            use_rag: Whether to use RAG for context
            use_kg: Whether to use knowledge graph
            
        Returns:
                Dictionary with analysis results
        """
        logger.info(f"Analyzing claim or URL: {claim}")
        start_time = datetime.now()
        
        # Check cache first
        cache_key = f"analysis_{hash(claim)}_{use_rag}_{use_kg}"
        if cache_key in result_cache:
            logger.info(f"Returning cached analysis for: {claim}")
            return result_cache[cache_key]
        
        try:
            # Check if input is a URL
            is_url_input = self.url_handler.is_url(claim)
            article_from_url = None
            source_credibility = None
            
            # If it's a URL, extract the article
            if is_url_input:
                logger.info(f"Input is a URL. Extracting article content...")
                article_from_url = self.url_handler.extract_article(claim)
                claim_for_analysis = article_from_url["content"]  # Use article content as the claim
                
                # If extraction failed, use the URL as the claim
                if not claim_for_analysis:
                    claim_for_analysis = claim
                    logger.warning(f"Failed to extract article content from URL: {claim}")
                
                # Analyze source credibility for URL inputs
                logger.info("Analyzing source credibility...")
                credibility_analysis = self.credibility_analyzer.assess_overall_source_credibility(
                    domain=article_from_url["source_domain"],
                    author=article_from_url["author"],
                    date_str=article_from_url["published_date"],
                    claim_text=claim_for_analysis
                )
                source_credibility = credibility_analysis.get("overall_credibility_score")
            else:
                claim_for_analysis = claim
            
            # Step 1: Extract entities
            logger.info("Extracting entities...")
            entities = self.entity_extractor.get_key_entities(claim_for_analysis)
            
            # Step 2: Analyze sentiment and emotional manipulation
            logger.info("Analyzing sentiment...")
            sentiment_analysis = self.sentiment_analyzer.detect_emotional_manipulation(claim_for_analysis)
            
            retrieved_articles = []
            if use_rag:
                # Step 3: Search for relevant articles
                logger.info("Searching for relevant articles...")
                articles = self.search_engine.search(claim_for_analysis)
                
                if not articles:
                    logger.warning("No articles found for the claim")
                else:
                    logger.info(f"Found {len(articles)} articles")
                    
                    # Step 4: Create embeddings
                    logger.info("Creating embeddings...")
                    texts = [article["full_text"] for article in articles]
                    claim_embedding = self.embedding_engine.get_embeddings([claim_for_analysis])
                    article_embeddings = self.embedding_engine.get_embeddings(texts)
                    
                    # Step 5: Calculate similarities and filter
                    similarities = self.embedding_engine.compute_similarity(claim_embedding[0], article_embeddings)
                    retrieved_articles = self.embedding_engine.filter_by_threshold(articles, similarities)
                    
                    logger.info(f"Retrieved {len(retrieved_articles)} relevant articles")
            
            # Step 6: Verify entities using Knowledge Graph
            entity_verification = {"status": "skipped"}
            if use_kg and self.knowledge_graph and self.knowledge_graph.available and entities:
                logger.info("Verifying entities with Knowledge Graph...")
                entity_verification = self.knowledge_graph.verify_entity_relationships(entities)
            
            # Step 7: Enhanced fact-check using Spark LLM with all our improvements
            logger.info("Enhanced fact-checking with Spark LLM...")
            fact_check_result = self.spark_llm.fact_check(claim_for_analysis, retrieved_articles, entities, source_credibility)
            
            if not fact_check_result:
                return {
                    "status": "error",
                    "message": "Failed to get fact-checking result from Spark LLM",
                    "claim": claim,
                    "sentiment_analysis": sentiment_analysis
                }
            
            # Step 8: Calibrate confidence score
            evidence_text = self.spark_llm.preprocess_evidence(retrieved_articles)
            calibrated_result = self.spark_llm.calibrate_confidence(
                fact_check_result, 
                claim_for_analysis, 
                evidence_text, 
                source_credibility, 
                entities
            )
            
            # New Step: Analyze credibility if URL input
            credibility_analysis = None
            
            if is_url_input and article_from_url:
                logger.info("Analyzing source credibility...")
                credibility_analysis = self.credibility_analyzer.assess_overall_source_credibility(
                    domain=article_from_url["source_domain"],
                    author=article_from_url["author"],
                    date_str=article_from_url["published_date"],
                    claim_text=claim_for_analysis
                )
            
            # Enhanced AI detection with improved output format
            logger.info("Performing AI content detection...")
            ai_detection_result = self.ai_detector.detect_ai_content(
                claim_for_analysis if not is_url_input else article_from_url["content"]
            )
            
            # Analyze title-content contradiction for URL inputs
            title_content_contradiction = None
            if is_url_input and article_from_url:
                try:
                    title_content_contradiction = self.spark_llm.detect_title_content_contradiction(
                        title=article_from_url['title'], 
                        content=article_from_url['content']
                    )
                except Exception as e:
                    logger.error(f"Error detecting title-content contradiction: {e}")
            
            # Step 9: Calculate overall trust lens score
            trust_lens_score = None
            if credibility_analysis:
                source_cred = credibility_analysis["overall_credibility_score"] 
                fact_confidence = calibrated_result.get("confidence", 0) / 100.0
                sentiment_score = sentiment_analysis.get("manipulation_score", 0)
                tone_neutrality = 1.0 - sentiment_score
                
                author_credibility = credibility_analysis.get("factors", {}).get("author", {}).get("credibility_factor", 0.5)
                
                # Calculate trust score
                trust_lens_score = self.credibility_analyzer.calculate_trust_lens_score(
                    source_credibility=source_cred,
                    factual_match=fact_confidence,
                    tone_neutrality=tone_neutrality,
                    source_transparency=author_credibility
                )
            
            # Step 10: Compile and return results
            result = {
                "status": "success",
                "claim": claim,
                "is_url_input": is_url_input,
                "entities": entities,
                "retrieved_articles": retrieved_articles,
                "sentiment_analysis": sentiment_analysis,
                "entity_verification": entity_verification,
                "fact_check": calibrated_result,
                "verdict": calibrated_result.get("verdict"),
                "confidence": calibrated_result.get("confidence"),
                "explanation": calibrated_result.get("explanation"),
                "reasoning": calibrated_result.get("reasoning"),
                "confidence_factors": calibrated_result.get("confidence_factors"),
                "ai_detection": ai_detection_result,  # Enhanced AI detection result
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
            
            # Add source metadata and credibility if URL input
            if is_url_input and article_from_url:
                result["source_metadata"] = {
                    "title": article_from_url["title"],
                    "author": article_from_url["author"],
                    "published_date": article_from_url["published_date"],
                    "domain": article_from_url["source_domain"]
                }
                result["credibility_analysis"] = credibility_analysis
            
        # Add title-content contradiction analysis if available
            if title_content_contradiction:
                result['title_content_contradiction'] = title_content_contradiction
            
            # Add trust lens score if available
            if trust_lens_score is not None:
                result["trust_lens_score"] = trust_lens_score
            
            # Add multi-pass verification results if available
            if "verification_analysis" in calibrated_result:
                result["verification_analysis"] = calibrated_result["verification_analysis"]
                
            if "verification_note" in calibrated_result:
                result["verification_note"] = calibrated_result["verification_note"]
            
            # Cache the result
            result_cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing claim: {e}")
            return {
                "status": "error",
                "message": str(e),
                "claim": claim
            } 
            
               
    def suggest_prompt(self, prompt: str) -> Dict[str, Any]:
        """Analyze a user's prompt and provide suggestions
        
        Args:
            prompt: The user's input prompt
            
        Returns:
            Dictionary with suggestions
        """
        return self.prompt_suggester.analyze_prompt_quality(prompt)

    def detect_ai_content(self, text: str) -> Dict[str, Any]:
        """Detect if content is likely AI-generated and categorize it
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with detection results
        """
        try:
            logger.info("Performing enhanced AI content detection...")
            ai_detection = self.ai_detector.detect_ai_content(text)
            content_category = self.ai_detector.categorize_content(text)
            
            # Combine results and ensure consistent output format
            result = {
                **ai_detection,
                "content_category": content_category.get("category", "unknown"),
                "content_category_scores": content_category.get("category_scores", {})
            }
            
            return result
        except Exception as e:
            logger.error(f"Error in AI content detection: {e}")
            return {
                "ai_score": 0.5,
                "ai_verdict": "Detection failed",
                "content_category": "unknown",
                "emoji": "❓",
                "display_message": "Could not determine if content was written by AI or human",
                "is_ai": None,
                "error_message": str(e)
            }
    
    def close(self):
        """Close any open connections"""
        if self.knowledge_graph:
            self.knowledge_graph.close()
            
    

# API Models for Request/Response validation
class AnalysisRequest(BaseModel):
    claim: str = Field(..., description="The claim to analyze or a URL to an article")
    use_rag: bool = Field(True, description="Whether to use RAG for context retrieval")
    use_kg: bool = Field(True, description="Whether to use knowledge graph for entity verification")
    is_url: bool = Field(False, description="Flag to indicate if the claim is a URL (auto-detected if not specified)")

class AnalysisResponse(BaseModel):
    status: str
    claim: str
    is_url_input: Optional[bool] = False
    verdict: Optional[str] = None
    confidence: Optional[int] = None
    explanation: Optional[str] = None
    emotional_manipulation: Optional[dict] = None
    credibility_assessment: Optional[dict] = None
    source_metadata: Optional[dict] = None
    processing_time: Optional[float] = None
    entities: Optional[dict] = None
    message: Optional[str] = None
    ai_detection: Optional[dict] = None  # Add this field!
    reasoning: Optional[str] = None
    trust_lens_score: Optional[float] = None
    title_content_contradiction: Optional[dict] = None

class PromptAnalysisRequest(BaseModel):
    prompt: str = Field(..., description="The user prompt to analyze")

class PromptAnalysisResponse(BaseModel):
    is_good_prompt: bool
    quality_score: float
    is_url: bool
    suggestions: List[str]
    improved_prompt: str





# API Server
app = FastAPI(
    title="TruthLens API",
    description="Fake news detection API using XLM ROBERTA with RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instance of the TruthLens system
truthlens = None

@app.on_event("startup")
async def startup_event():
    global truthlens
    try:
        truthlens = FakeNewsDetectionSystem()
        logger.info("TruthLens system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize TruthLens system: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    global truthlens
    if truthlens:
        truthlens.close()
        logger.info("TruthLens system shut down")

@app.get("/")
async def root():
    return {"message": "Welcome to TruthLens API - Fake News Detection with Spark LLM"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": str(datetime.now())
    }



# Possible fixes for the PromptQualityAnalyzer in Try_train.py

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_claim(request: AnalysisRequest):
    """
    Analyze a claim or URL for fake news detection.
    """
    global truthlens
    
    if not truthlens:
        raise HTTPException(status_code=503, detail="TruthLens system not initialized")
    
    try:
        result = truthlens.analyze_claim(
            claim=request.claim,
            use_rag=request.use_rag,
            use_kg=request.use_kg
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        
        # Extract relevant information for the response
        response = {
            "status": result["status"],
            "claim": result["claim"],
            "is_url_input": result.get("is_url_input", False),
            "verdict": None,  # We'll set this below
            "confidence": None,
            "explanation": None,
            "emotional_manipulation": {
                "score": result["sentiment_analysis"]["manipulation_score"],
                "level": result["sentiment_analysis"]["manipulation_level"],
                "explanation": result["sentiment_analysis"]["explanation"],
                "propaganda_analysis": result["sentiment_analysis"].get("details", {}).get("propaganda_analysis")
            },
            "processing_time": result["processing_time"],
            "entities": result["entities"]
        }
        
        # Important: Make sure AI detection is explicitly included
        if "ai_detection" in result:
            response["ai_detection"] = result["ai_detection"]
            print(f"AI detection data included in response: {result['ai_detection'] is not None}")
        else:
            print("No AI detection data found in analysis result")
        
        # Parse fact_check result if it's a string
        if isinstance(result["fact_check"], str):
            fact_check_text = result["fact_check"]
            
            # Extract verdict with improved regex
            verdict_match = re.search(r"Verdict:\s*(.*?)(?:\n|$)", fact_check_text, re.IGNORECASE)
            if verdict_match:
                response["verdict"] = verdict_match.group(1).strip()
            else:
                # Try alternate patterns for verdict extraction
                alt_verdict_match = re.search(r"(True|False|Partially True|REAL|FAKE|MISLEADING)", fact_check_text)
                if alt_verdict_match:
                    response["verdict"] = alt_verdict_match.group(1).strip()
                else:
                    # If we can't extract verdict, use a default based on confidence
                    if "confidence" in response and response["confidence"]:
                        conf = response["confidence"]
                        if conf >= 80:
                            response["verdict"] = "True"
                        elif conf >= 40:
                            response["verdict"] = "Partially True"
                        else:
                            response["verdict"] = "False"
                    else:
                        response["verdict"] = "Unverified"
            
            # Extract confidence
            confidence_match = re.search(r"Confidence:\s*(\d+)%", fact_check_text)
            if confidence_match:
                response["confidence"] = int(confidence_match.group(1))
            else:
                # Try to find any number followed by % sign if the standard format fails
                alt_confidence_match = re.search(r"(\d+)%", fact_check_text)
                if alt_confidence_match:
                    response["confidence"] = int(alt_confidence_match.group(1))
                else:
                    # Default confidence if we can't extract it
                    response["confidence"] = 75
            
            # Extract explanation
            explanation_match = re.search(r"Explanation:\s*(.*?)(?:\n\n|$)", fact_check_text, re.DOTALL)
            if explanation_match:
                response["explanation"] = explanation_match.group(1).strip()
            else:
                # Take a larger chunk as explanation if the format doesn't match
                explanation_fallback = "\n".join(fact_check_text.split("\n")[3:])
                response["explanation"] = explanation_fallback[:500] + "..."
                
        else:
            # Handle as dictionary if it's not a string
            response["verdict"] = result["fact_check"].get("verdict")
            response["confidence"] = result["fact_check"].get("confidence")
            response["explanation"] = result["fact_check"].get("explanation")
        
        # Add source metadata if available
        if "source_metadata" in result:
            response["source_metadata"] = result["source_metadata"]
        
        # Add credibility assessment if available
        if "credibility_analysis" in result:
            response["credibility_assessment"] = {
                "score": result["credibility_analysis"]["overall_credibility_score"],
                "level": result["credibility_analysis"]["credibility_level"],
                "explanation": result["credibility_analysis"]["explanation"],
                "details": result["credibility_analysis"]["details"]
            }
            
            # For URL inputs, if verdict is still None, derive it from credibility assessment
            if response["verdict"] is None and response["is_url_input"]:
                cred_score = result["credibility_analysis"]["overall_credibility_score"]
                if cred_score >= 0.75:
                    response["verdict"] = "True"
                elif cred_score >= 0.4:
                    response["verdict"] = "Partially True"
                else:
                    response["verdict"] = "False"
        
        # If verdict is still None or "Unknown", provide a default based on the text assessment
        if response["verdict"] is None or response["verdict"] == "Unknown":
            # Check if there's any indication in the explanation
            if response["explanation"] and "true" in response["explanation"].lower():
                response["verdict"] = "True"
            elif response["explanation"] and "false" in response["explanation"].lower():
                response["verdict"] = "False"
            elif response["explanation"] and "partially" in response["explanation"].lower():
                response["verdict"] = "Partially True"
            else:
                # Final fallback
                response["verdict"] = "Unverified"
                
        # Add additional analysis data
        if 'title_content_contradiction' in result:
            response['title_content_contradiction'] = result['title_content_contradiction']
            
        # Add the reasoning field if available
        if "reasoning" in result["fact_check"]:
            response["reasoning"] = result["fact_check"]["reasoning"]
        
        # Calculate trust_lens_score
        if 'credibility_analysis' in result and 'overall_credibility_score' in result['credibility_analysis']:
           source_credibility = result["credibility_analysis"]["overall_credibility_score"] 
           fact_confidence = response.get("confidence", 0) / 100.0 if response.get("confidence") else 0.5
           sentiment_score = result["sentiment_analysis"].get("manipulation_score", 0)
           tone_neutrality = 1.0 - sentiment_score
           
           author_credibility = result.get("credibility_analysis", {}).get("factors", {}).get("author", {}).get("credibility_factor", 0.5)
           
           # Calculate trust score
           trust_lens_score = truthlens.credibility_analyzer.calculate_trust_lens_score(
                source_credibility=source_credibility,
                factual_match=fact_confidence,
                tone_neutrality=tone_neutrality,
                source_transparency=author_credibility
            )
           response["trust_lens_score"] = trust_lens_score
        
        # Log the final verdict and confidence
        logger.info(f"Final verdict: {response['verdict']}, Confidence: {response['confidence']}%")
        
        # Log AI detection status
        if "ai_detection" in response:
            logger.info("AI detection included in response")
        else:
            logger.warning("AI detection missing from response")
        
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing analysis request: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze-prompt", response_model=PromptAnalysisResponse)
async def analyze_prompt(request: PromptAnalysisRequest):
    """
    Analyze a user's prompt and provide suggestions for improvement.
    """
    global truthlens
    
    if not truthlens:
        raise HTTPException(status_code=503, detail="TruthLens system not initialized")
    
    try:
        result = truthlens.suggest_prompt(request.prompt)
        return result
    except Exception as e:
        logger.error(f"Error analyzing prompt: {e}")
        raise HTTPException(status_code=500, detail=f"Prompt analysis failed: {str(e)}")




# Command line interface
def main():
    """Command line interface for TruthLens"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TruthLens: Fake News Detection with Spark LLM")
    parser.add_argument("--claim", type=str, help="Claim to analyze or URL to analyze")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG")
    parser.add_argument("--no-kg", action="store_true", help="Disable Knowledge Graph")
    parser.add_argument("--api", action="store_true", help="Run as API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    
    args = parser.parse_args()
    
    if args.api:
        # Run as API server
        uvicorn.run("Try_train:app", host=args.host, port=args.port, reload=True)
    elif args.claim:
        # Run in CLI mode
        system = FakeNewsDetectionSystem()
        
        try:
            result = system.analyze_claim(
                claim=args.claim,
                use_rag=not args.no_rag,
                use_kg=not args.no_kg
            )
            
            if result["status"] == "success":
                print("\n=== ANALYSIS RESULTS ===")
                print(f"Claim: {result['claim']}")
                
                # Print if input was URL
                if result.get("is_url_input", False):
                    print("\nInput Type: URL")
                    if "source_metadata" in result:
                        print("\nSource Metadata:")
                        print(f"- Title: {result['source_metadata']['title']}")
                        print(f"- Author: {result['source_metadata']['author']}")
                        print(f"- Date: {result['source_metadata']['published_date']}")
                        print(f"- Domain: {result['source_metadata']['domain']}")
                
                # Print credibility if available
                if "credibility_analysis" in result:
                    print("\nSource Credibility Assessment:")
                    print(f"- Score: {result['credibility_analysis']['overall_credibility_score']:.2f}/1.0")
                    print(f"- Level: {result['credibility_analysis']['credibility_level']}")
                    print(f"- Summary: {result['credibility_analysis']['explanation']}")
                    print("\nCredibility Details:")
                    for detail in result['credibility_analysis']['details']:
                        print(f"  {detail}")
                
                print("\nEntities Detected:")
                for entity_type, entities in result['entities'].items():
                    print(f"- {entity_type}: {', '.join(entities)}")
                
                print("\nSentiment Analysis:")
                print(f"Emotional Manipulation: {result['sentiment_analysis']['manipulation_score']:.2f} ({result['sentiment_analysis']['manipulation_level']})")
                print(f"Dominant Emotion: {result['sentiment_analysis']['dominant_emotion']}")
                print(f"Explanation: {result['sentiment_analysis']['explanation']}")
                
                # Extract fact check information
                if isinstance(result["fact_check"], str):
                    fact_check_text = result["fact_check"]
                    
                    # Extract verdict
                    verdict_match = re.search(r"Verdict: (.*?)(?:\n|$)", fact_check_text)
                    verdict = verdict_match.group(1).strip() if verdict_match else "Unknown"
                    
                    # Extract confidence
                    confidence_match = re.search(r"Confidence: (\d+)%", fact_check_text)
                    confidence = confidence_match.group(1) if confidence_match else "Unknown"
                    
                    # Extract explanation
                    explanation_match = re.search(r"Explanation: (.*?)(?:\n\n|$)", fact_check_text, re.DOTALL)
                    explanation = explanation_match.group(1).strip() if explanation_match else "No explanation available"
                    
                    print("\nFact Check Result:")
                    print(f"Verdict: {verdict} (Confidence: {confidence}%)")
                    print(f"Explanation: {explanation}")
                    
                    # Print raw fact check for detailed reasoning
                    print("\nDetailed Reasoning:")
                    print(fact_check_text)
                else:
                    # Handle if it's an object
                    print("\nFact Check Result:")
                    print(f"Verdict: {result['fact_check']['verdict']} (Confidence: {result['fact_check']['confidence']}%)")
                    print(f"Explanation: {result['fact_check']['explanation']}")
                
                print(f"\nProcessing Time: {result['processing_time']:.2f} seconds")
                print("=" * 50)
            else:
                print(f"Error: {result['message']}")
                
        except Exception as e:
            print(f"Error: {e}")
        finally:
            system.close()
    else:
        parser.print_help()
    

if __name__ == "__main__":
    main()

