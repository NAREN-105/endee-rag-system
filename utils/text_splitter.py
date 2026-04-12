from typing import List, Dict
import re

class TextSplitter:
    """Advanced text splitting with overlap and metadata preservation"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = chunk_overlap
    
    def split_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into chunks with overlap
        
        Args:
            text: Input text to split
            metadata: Additional metadata to attach to chunks
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or not text.strip():
            return []
        
        # Clean text
        text = self._clean_text(text)
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence exceeds chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, len(chunks), metadata))
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, len(chunks), metadata))
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\']', '', text)
        return text.strip()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for overlap"""
        overlap_chars = 0
        overlap_sentences = []
        
        for sentence in reversed(sentences):
            if overlap_chars >= self.overlap:
                break
            overlap_sentences.insert(0, sentence)
            overlap_chars += len(sentence)
        
        return overlap_sentences
    
    def _create_chunk(self, text: str, index: int, metadata: Dict = None) -> Dict:
        """Create chunk dictionary with metadata"""
        chunk = {
            "text": text,
            "chunk_index": index,
            "char_count": len(text),
            "word_count": len(text.split())
        }
        
        if metadata:
            chunk.update(metadata)
        
        return chunk
    # Alias for compatibility
SmartTextSplitter = TextSplitter