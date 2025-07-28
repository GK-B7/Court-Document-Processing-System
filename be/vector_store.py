"""
Vector Store for Action Matching using ChromaDB
──────────────────────────────────────────────
Semantic search and matching for document actions using only your 2 database actions:
- freeze_funds: Freeze all funds in the customer account
- release_funds: Release all held or restricted funds back to the customer account

Features:
• ChromaDB integration for semantic similarity
• Action matching with confidence scoring
• Handles only freeze_funds, release_funds, or returns no match
• Simplified for your exact database schema
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import json

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import settings
from database import db_manager
from psycopg2.extras import RealDictCursor
from exceptions import VectorStoreError
from metrics import (
    increment_database_query,
    record_database_query_time,
)

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration and Constants
# ─────────────────────────────────────────────────────────────────────────────

# ChromaDB Configuration
CHROMA_CONFIG = {
    'persist_directory': getattr(settings, 'CHROMA_PERSIST_DIRECTORY', './chroma_db'),
    'collection_name': 'banking_actions',
    'embedding_model': 'all-MiniLM-L6-v2',  # Lightweight model
    'similarity_threshold': getattr(settings, 'SIMILARITY_THRESHOLD', 0.7),
    'max_results': 3
}

# Your 2 actions with enhanced examples for better matching
ACTION_DEFINITIONS = {
    'freeze_funds': {
        'description': 'Freeze all funds in the customer account',
        'examples': [
            'freeze funds',
            'freeze account',
            'hold funds',
            'block funds',
            'restrict funds',
            'suspend funds',
            'lock account',
            'freeze customer funds',
            'put hold on funds',
            'restrict account access',
            'block account funds',
            'freeze all funds',
            'hold customer funds',
            'suspend account funds'
        ],
        'keywords': ['freeze', 'hold', 'block', 'restrict', 'suspend', 'lock']
    },
    'release_funds': {
        'description': 'Release all held or restricted funds back to the customer account',
        'examples': [
            'release funds',
            'unfreeze funds',
            'release held funds',
            'unblock funds',
            'restore funds',
            'release account',
            'unfreeze account',
            'remove hold',
            'lift restrictions',
            'restore account access',
            'release customer funds',
            'unblock account funds',
            'remove fund hold',
            'restore normal access'
        ],
        'keywords': ['release', 'unfreeze', 'unblock', 'restore', 'remove', 'lift']
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB Client and Collection Management
# ─────────────────────────────────────────────────────────────────────────────

class ActionVectorStore:
    """Vector store for banking action matching using only your 2 actions"""
    
    def __init__(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=CHROMA_CONFIG['persist_directory']
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=CHROMA_CONFIG['collection_name'],
                metadata={"description": "Banking actions for freeze_funds and release_funds only"}
            )
            
            # Initialize sentence transformer for embeddings
            self.embedding_model = SentenceTransformer(CHROMA_CONFIG['embedding_model'])
            
            # Statistics
            self.stats = {
                'total_queries': 0,
                'successful_matches': 0,
                'no_matches': 0,
                'total_query_time': 0.0,
                'cache_hits': 0
            }
            
            logger.info("ActionVectorStore initialized successfully")
            
        except Exception as exc:
            logger.error(f"Failed to initialize ActionVectorStore: {exc}")
            raise VectorStoreError(
                f"Vector store initialization failed: {exc}",
                operation="initialization",
                original_exception=exc
            ) from exc
    
    def initialize_from_database(self) -> None:
        """Initialize vector store with your 2 actions from database"""
        try:
            # Fix: Clear existing data properly
            try:
                # Get all IDs first, then delete them
                existing_items = self.collection.get()
                if existing_items['ids']:
                    self.collection.delete(ids=existing_items['ids'])
                    logger.info("Cleared existing action data")
                else:
                    logger.info("No existing data to clear")
            except Exception as clear_exc:
                logger.warning(f"Could not clear existing data: {clear_exc}")
                # Continue anyway - collection might be empty
            
            # Load actions from your database
            db_actions = self._load_actions_from_database()
            
            if not db_actions:
                logger.warning("No actions found in database, using hardcoded definitions")
                db_actions = [
                    {'action_name': 'freeze_funds', 'description': 'Freeze all funds in the customer account'},
                    {'action_name': 'release_funds', 'description': 'Release all held or restricted funds back to the customer account'}
                ]
            
            # Add each action with examples to ChromaDB
            for db_action in db_actions:
                action_name = db_action['action_name']
                
                if action_name in ACTION_DEFINITIONS:
                    self._add_action_to_collection(
                        action_name=action_name,
                        description=db_action['description'],
                        examples=ACTION_DEFINITIONS[action_name]['examples']
                    )
                else:
                    logger.warning(f"Unknown action in database: {action_name}")
            
            logger.info(f"Initialized vector store with {len(db_actions)} actions from database")
            
        except Exception as exc:
            logger.error(f"Failed to initialize from database: {exc}")
            # Don't raise error - continue with hardcoded actions
            self._initialize_with_defaults()

    def _initialize_with_defaults(self) -> None:
        """Initialize with hardcoded actions as fallback"""
        try:
            logger.info("Initializing with default actions")
            
            for action_name, action_def in ACTION_DEFINITIONS.items():
                self._add_action_to_collection(
                    action_name=action_name,
                    description=action_def['description'],
                    examples=action_def['examples']
                )
            
            logger.info("Initialized with default actions successfully")
            
        except Exception as exc:
            logger.error(f"Failed to initialize with defaults: {exc}")
            # This is a critical error - can't continue without actions
            raise VectorStoreError(
                f"Failed to initialize action store: {exc}",
                operation="default_initialization",
                original_exception=exc
            ) from exc

    
    def _load_actions_from_database(self) -> List[Dict[str, Any]]:
        """Load actions from your database actions table"""
        try:
            increment_database_query()
            start_time = time.perf_counter()
            
            with db_manager.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT action_name, description 
                        FROM actions 
                        ORDER BY action_name
                    """)
                    
                    results = cursor.fetchall()
                    actions = [dict(row) for row in results]
                    
                    query_time = time.perf_counter() - start_time
                    record_database_query_time(query_time)
                    
                    logger.info(f"Loaded {len(actions)} actions from database")
                    return actions
                    
        except Exception as exc:
            logger.error(f"Failed to load actions from database: {exc}")
            return []
    
    def _add_action_to_collection(
        self, 
        action_name: str, 
        description: str, 
        examples: List[str]
    ) -> None:
        """Add action with examples to ChromaDB collection"""
        try:
            # Create documents for the action (description + examples)
            documents = [description] + examples
            
            # Generate IDs
            ids = [f"{action_name}_desc"] + [f"{action_name}_ex_{i}" for i in range(len(examples))]
            
            # Create metadata
            metadatas = [
                {
                    'action_name': action_name,
                    'type': 'description',
                    'text': description
                }
            ] + [
                {
                    'action_name': action_name,
                    'type': 'example',
                    'text': example
                }
                for example in examples
            ]
            
            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.debug(f"Added action '{action_name}' with {len(examples)} examples")
            
        except Exception as exc:
            logger.error(f"Failed to add action {action_name}: {exc}")
            raise VectorStoreError(
                f"Failed to add action to collection: {exc}",
                operation="add_action",
                collection_name=CHROMA_CONFIG['collection_name'],
                original_exception=exc
            ) from exc
    
    def find_best_action_match(
        self, 
        query_text: str, 
        threshold: float = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find best matching action for query text
        Returns only freeze_funds, release_funds, or None
        """
        start_time = time.perf_counter()
        
        if threshold is None:
            threshold = CHROMA_CONFIG['similarity_threshold']
        
        try:
            self.stats['total_queries'] += 1
            
            # Clean and normalize query
            query_text = query_text.strip().lower()
            
            if not query_text:
                self.stats['no_matches'] += 1
                return None
            
            # Query ChromaDB for similar actions
            results = self.collection.query(
                query_texts=[query_text],
                n_results=CHROMA_CONFIG['max_results'],
                include=['documents', 'distances', 'metadatas']
            )
            
            # Process results
            best_match = self._process_query_results(results, threshold)
            
            # Update statistics
            query_time = time.perf_counter() - start_time
            self.stats['total_query_time'] += query_time
            
            if best_match:
                self.stats['successful_matches'] += 1
                logger.debug(
                    f"Action match found: {best_match['action_name']} "
                    f"(similarity: {best_match['similarity']:.3f})"
                )
            else:
                self.stats['no_matches'] += 1
                logger.debug(f"No action match found for: '{query_text}'")
            
            return best_match
            
        except Exception as exc:
            query_time = time.perf_counter() - start_time
            self.stats['total_query_time'] += query_time
            
            logger.error(f"Action matching failed for '{query_text}': {exc}")
            raise VectorStoreError(
                f"Action matching failed: {exc}",
                operation="query",
                query_text=query_text,
                original_exception=exc
            ) from exc
    
    def _process_query_results(
        self, 
        results: Dict[str, Any], 
        threshold: float
    ) -> Optional[Dict[str, Any]]:
        """Process ChromaDB query results and return best match"""
        if not results['distances'] or not results['distances'][0]:
            return None
        
        # Find best match
        best_distance = results['distances'][0][0]
        best_similarity = 1 - best_distance
        
        # Check if similarity meets threshold
        if best_similarity < threshold:
            return None
        
        # Get best match metadata
        best_metadata = results['metadatas'][0][0]
        best_document = results['documents'][0][0]
        action_name = best_metadata['action_name']
        
        # Only return if it's one of your 2 supported actions
        if action_name not in ['freeze_funds', 'release_funds']:
            return None
        
        return {
            'action_name': action_name,
            'similarity': best_similarity,
            'confidence': min(best_similarity, 1.0),
            'description': ACTION_DEFINITIONS[action_name]['description'],
            'matched_text': best_document,
            'metadata': {
                'match_type': best_metadata['type'],
                'distance': best_distance,
                'threshold_used': threshold
            }
        }
    
    def get_action_info(self, action_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific action"""
        if action_name not in ACTION_DEFINITIONS:
            return None
        
        action_def = ACTION_DEFINITIONS[action_name]
        return {
            'action_name': action_name,
            'description': action_def['description'],
            'examples': action_def['examples'][:5],  # First 5 examples
            'keywords': action_def['keywords']
        }
    
    def get_all_actions(self) -> List[Dict[str, Any]]:
        """Get information about all supported actions"""
        return [
            self.get_action_info(action_name) 
            for action_name in ACTION_DEFINITIONS.keys()
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        total_queries = self.stats['total_queries']
        avg_query_time = (
            self.stats['total_query_time'] / total_queries 
            if total_queries > 0 else 0
        )
        
        return {
            **self.stats,
            'average_query_time': avg_query_time,
            'match_rate': (
                self.stats['successful_matches'] / total_queries 
                if total_queries > 0 else 0
            ),
            'collection_count': self.collection.count(),
            'supported_actions': list(ACTION_DEFINITIONS.keys()),
            'similarity_threshold': CHROMA_CONFIG['similarity_threshold']
        }
    
    def reset_stats(self) -> None:
        """Reset statistics"""
        self.stats = {
            'total_queries': 0,
            'successful_matches': 0,
            'no_matches': 0,
            'total_query_time': 0.0,
            'cache_hits': 0
        }


# ─────────────────────────────────────────────────────────────────────────────
# Global Instance and Utility Functions
# ─────────────────────────────────────────────────────────────────────────────

# Global vector store instance
action_store = ActionVectorStore()

def initialize_action_store() -> None:
    """Initialize the action store with your database actions"""
    try:
        action_store.initialize_from_database()
        logger.info("Action store initialized successfully")
    except Exception as exc:
        logger.error(f"Failed to initialize action store: {exc}")
        raise

def get_best_action_match(
    query_text: str, 
    threshold: float = None
) -> Optional[Dict[str, Any]]:
    """
    Convenience function to find best action match
    Returns freeze_funds, release_funds, or None
    """
    return action_store.find_best_action_match(query_text, threshold)

def get_supported_actions() -> List[str]:
    """Get list of supported action names"""
    return list(ACTION_DEFINITIONS.keys())

def get_action_examples(action_name: str) -> List[str]:
    """Get examples for a specific action"""
    if action_name in ACTION_DEFINITIONS:
        return ACTION_DEFINITIONS[action_name]['examples']
    return []

def search_actions_by_keyword(keyword: str) -> List[str]:
    """Find actions that match a keyword"""
    keyword = keyword.lower().strip()
    matching_actions = []
    
    for action_name, action_def in ACTION_DEFINITIONS.items():
        if keyword in action_def['keywords']:
            matching_actions.append(action_name)
    
    return matching_actions

def get_vector_store_stats() -> Dict[str, Any]:
    """Get vector store statistics"""
    return action_store.get_stats()

def test_action_matching() -> None:
    """Test action matching with sample queries"""
    test_queries = [
        "freeze customer funds",
        "hold the money",
        "release funds back to customer", 
        "unfreeze account",
        "block funds",
        "restore account access",
        "suspend account",
        "remove restrictions",
        "unknown action here",
        "close account"  # This should return None
    ]
    
    print("Testing Action Matching:")
    print("=" * 50)
    
    for query in test_queries:
        result = get_best_action_match(query)
        if result:
            print(f"Query: '{query}'")
            print(f"  -> Match: {result['action_name']} (similarity: {result['similarity']:.3f})")
        else:
            print(f"Query: '{query}'")
            print(f"  -> No match found")
        print()

# Initialize on import (can be called again if needed)
try:
    initialize_action_store()
except Exception as e:
    logger.warning(f"Initial action store setup failed: {e}")

# Export main functions and classes
__all__ = [
    'ActionVectorStore',
    'action_store', 
    'initialize_action_store',
    'get_best_action_match',
    'get_supported_actions',
    'get_action_examples',
    'search_actions_by_keyword',
    'get_vector_store_stats',
    'test_action_matching'
]
