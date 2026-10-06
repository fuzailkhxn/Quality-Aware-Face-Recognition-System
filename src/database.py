"""
Face Database Module
Manages registration and storage of face embeddings.
"""

import os
import pickle
import numpy as np


# Default paths
DEFAULT_REGISTERED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset", "registered"
)
DEFAULT_EMBEDDINGS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "embeddings.pkl"
)


class FaceDatabase:
    """Manages face embeddings for registered persons."""
    
    def __init__(self, embeddings_file=None):
        self.embeddings_file = embeddings_file or DEFAULT_EMBEDDINGS_FILE
        self.database = {}  # {name: [embedding1, embedding2, ...]}
        self.load()
    
    def load(self):
        """Load embeddings from file."""
        if os.path.exists(self.embeddings_file):
            try:
                with open(self.embeddings_file, 'rb') as f:
                    self.database = pickle.load(f)
                print(f"[INFO] Loaded {len(self.database)} identities from database")
            except Exception as e:
                print(f"[WARN] Could not load database: {e}")
                self.database = {}
        else:
            self.database = {}
    
    def save(self):
        """Save embeddings to file."""
        os.makedirs(os.path.dirname(self.embeddings_file), exist_ok=True)
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(self.database, f)
        print(f"[INFO] Database saved to {self.embeddings_file}")
    
    def add_embedding(self, name, embedding):
        """Add an embedding for a person."""
        if name not in self.database:
            self.database[name] = []
        self.database[name].append(embedding)
    
    def get_all_identities(self):
        """Get list of all registered identities."""
        return list(self.database.keys())
    
    def get_embeddings(self, name):
        """Get all embeddings for a person."""
        return self.database.get(name, [])
    
    def get_average_embedding(self, name):
        """Get average embedding for a person."""
        embeddings = self.get_embeddings(name)
        if not embeddings:
            return None
        return np.mean(embeddings, axis=0)
    
    def is_empty(self):
        """Check if database has any registered faces."""
        return len(self.database) == 0
    
    def get_stats(self):
        """Get database statistics."""
        stats = {}
        for name, embeddings in self.database.items():
            stats[name] = len(embeddings)
        return stats
    
    def clear(self):
        """Clear all embeddings."""
        self.database = {}
        if os.path.exists(self.embeddings_file):
            os.remove(self.embeddings_file)
