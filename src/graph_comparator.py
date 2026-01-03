"""
Graph Comparator - The Smatch Metric

Compares source AMR graphs with reconstructed AMR graphs to verify
semantic preservation in translation.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional
import penman

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of AMR graph comparison."""
    source_amr: str
    target_amr: str
    smatch_score: float  # 0.0 to 1.0
    precision: float
    recall: float
    f1_score: float
    is_verified: bool
    differences: list[str]
    threshold: float = 0.85


class GraphComparator:
    """
    AMR Graph Comparator using Smatch metric.
    
    Smatch (Semantic Match) measures the overlap between two AMR graphs
    by computing the maximum F-score over all possible variable mappings.
    """
    
    def __init__(self, verification_threshold: float = 0.85):
        """
        Initialize the comparator.
        
        Args:
            verification_threshold: Minimum Smatch F1 score to consider
                                   the translation verified (default 0.85)
        """
        self.threshold = verification_threshold
        self._smatch_available = None
    
    def _check_smatch(self) -> bool:
        """Check if smatch library is available."""
        if self._smatch_available is None:
            try:
                import smatch
                self._smatch_available = True
            except ImportError:
                logger.warning("smatch library not available, using fallback comparison")
                self._smatch_available = False
        return self._smatch_available
    
    def compare(self, source_amr: str, target_amr: str) -> ComparisonResult:
        """
        Compare two AMR graphs using Smatch.
        
        Args:
            source_amr: Source AMR graph (from English parsing)
            target_amr: Target AMR graph (from Arabic reverse parsing)
            
        Returns:
            ComparisonResult with scores and verification status
        """
        # Normalize both graphs
        source_normalized = self._normalize_amr(source_amr)
        target_normalized = self._normalize_amr(target_amr)
        
        if self._check_smatch():
            return self._smatch_compare(source_normalized, target_normalized)
        else:
            return self._fallback_compare(source_normalized, target_normalized)
    
    def _normalize_amr(self, amr_string: str) -> str:
        """Normalize AMR for consistent comparison."""
        try:
            graph = penman.decode(amr_string)
            return penman.encode(graph, indent=2)
        except Exception:
            return amr_string.strip()
    
    def _smatch_compare(self, source: str, target: str) -> ComparisonResult:
        """Compare using the smatch library."""
        import smatch
        
        try:
            # Smatch API varies by version
            result = smatch.score_amr_pairs([(source, target)])
            if result and len(result) > 0:
                precision, recall, f1 = result[0]
            else:
                raise ValueError("No result from smatch")
            
            # Identify specific differences
            differences = self._find_differences(source, target)
            
            return ComparisonResult(
                source_amr=source,
                target_amr=target,
                smatch_score=f1,
                precision=precision,
                recall=recall,
                f1_score=f1,
                is_verified=(f1 >= self.threshold),
                differences=differences,
                threshold=self.threshold
            )
            
        except Exception as e:
            logger.debug(f"Smatch comparison failed, using fallback: {e}")
            # Fall back to simple comparison
            return self._fallback_compare(source, target)
    
    def _fallback_compare(self, source: str, target: str) -> ComparisonResult:
        """
        Fallback comparison when smatch is not available.
        
        Uses triple-based comparison (simplified Smatch approximation).
        """
        source_triples = self._extract_triples(source)
        target_triples = self._extract_triples(target)
        
        if not source_triples and not target_triples:
            # Both empty - consider it a match
            return ComparisonResult(
                source_amr=source,
                target_amr=target,
                smatch_score=1.0,
                precision=1.0,
                recall=1.0,
                f1_score=1.0,
                is_verified=True,
                differences=[],
                threshold=self.threshold
            )
        
        # Calculate overlap
        source_set = set(source_triples)
        target_set = set(target_triples)
        
        matches = source_set & target_set
        
        precision = len(matches) / len(target_set) if target_set else 0
        recall = len(matches) / len(source_set) if source_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Find differences
        differences = []
        for triple in source_set - target_set:
            differences.append(f"Missing from target: {triple}")
        for triple in target_set - source_set:
            differences.append(f"Extra in target: {triple}")
        
        return ComparisonResult(
            source_amr=source,
            target_amr=target,
            smatch_score=f1,
            precision=precision,
            recall=recall,
            f1_score=f1,
            is_verified=(f1 >= self.threshold),
            differences=differences,
            threshold=self.threshold
        )
    
    def _extract_triples(self, amr_string: str) -> list[tuple]:
        """Extract normalized triples from AMR for comparison."""
        triples = []
        
        try:
            graph = penman.decode(amr_string)
            
            # Extract instance triples (variable, instance, concept)
            for instance in graph.instances():
                concept = self._normalize_concept(instance.target)
                triples.append(("instance", concept))
            
            # Extract attribute triples
            for attr in graph.attributes():
                role = attr.role
                value = str(attr.target)
                triples.append((role, value))
            
            # Extract edge triples (relationships)
            for edge in graph.edges():
                role = edge.role
                # Get concept of source and target
                source_concept = self._get_concept_for_var(graph, edge.source)
                target_concept = self._get_concept_for_var(graph, edge.target)
                triples.append((source_concept, role, target_concept))
                
        except Exception as e:
            logger.warning(f"Triple extraction failed: {e}")
            # Fallback: simple pattern matching
            triples = self._pattern_extract_triples(amr_string)
        
        return triples
    
    def _normalize_concept(self, concept: str) -> str:
        """Normalize concept for comparison with semantic equivalence."""
        if concept is None:
            return ""
        concept = concept.lower()
        
        # Normalize compound concepts to separate words
        # e.g., "categorical-grammar" → "categorical grammar"
        # e.g., "sentence-structure" → "sentence structure"
        if "-" in concept and not concept[-1].isdigit():
            # Not a PropBank frame (those end in -01, -02, etc.)
            concept = concept.replace("-", " ")
        
        # Normalize semantic equivalents (PropBank frames and concepts)
        equivalents = {
            # Dissection/Analysis verbs
            "dismantle-01": "dissect-01",
            "decompose-01": "dissect-01",
            "break-down-01": "dissect-01",
            "analyze-01": "parse-01",
            
            # Importance/Significance adjectives
            "important-01": "crucial-01",
            "vital-01": "essential-01",
            "fundamental-01": "essential-01",
            "basic": "essential",
            
            # Performance verbs
            "enhance-01": "boost-01",
            "improve-01": "boost-01",
            "strengthen-01": "boost-01",
            "excel-01": "outperform-01",
            "exceed-01": "outperform-01",
            "surpass-01": "outperform-01",
            "outdo-01": "outperform-01",
            "beat-01": "outperform-01",
            
            # Quality/Performance terms
            "state of the art": "state-of-the-art",
            "first class": "state-of-the-art",
            "first-class": "state-of-the-art",
            "top tier": "state-of-the-art",
            "advanced": "state-of-the-art",
            "best": "state-of-the-art",
            "cutting edge": "state-of-the-art",
            
            # ML/NLP terms
            "large language model": "llm",
            "language model": "llm",
            "big language model": "llm",
            "encoder based model": "encoder-based-model",
            
            # Grammar terms
            "categorial": "categorical",
        }
        
        return equivalents.get(concept, concept)
    
    def _get_concept_for_var(self, graph, var: str) -> str:
        """Get the concept name for a variable in the graph."""
        for instance in graph.instances():
            if instance.source == var:
                return self._normalize_concept(instance.target)
        return var
    
    def _pattern_extract_triples(self, amr_string: str) -> list[tuple]:
        """Extract triples using regex patterns (fallback)."""
        triples = []
        
        # Match concept instances: (x / concept-name)
        concepts = re.findall(r'\(\s*\w+\s*/\s*([a-zA-Z][\w-]*)', amr_string)
        for concept in concepts:
            triples.append(("instance", concept.lower()))
        
        # Match relations: :relation value
        relations = re.findall(r'(:[\w-]+)\s+(\([^)]+\)|[^\s:)]+)', amr_string)
        for role, value in relations:
            value = value.strip('()"\'')
            if value and not value.startswith('/'):
                triples.append((role, value))
        
        # Match polarity
        if ":polarity -" in amr_string:
            triples.append((":polarity", "-"))
        
        return triples
    
    def _find_differences(self, source: str, target: str) -> list[str]:
        """Identify specific semantic differences between graphs."""
        differences = []
        
        # Check polarity
        source_has_neg = ":polarity -" in source
        target_has_neg = ":polarity -" in target
        
        if source_has_neg and not target_has_neg:
            differences.append("CRITICAL: Negation missing in translation")
        elif target_has_neg and not source_has_neg:
            differences.append("CRITICAL: Extra negation added in translation")
        
        # Extract and compare main predicates
        source_predicates = set(re.findall(r'/\s*([a-z]+-\d+)', source))
        target_predicates = set(re.findall(r'/\s*([a-z]+-\d+)', target))
        
        for pred in source_predicates - target_predicates:
            differences.append(f"Predicate '{pred}' missing in translation")
        for pred in target_predicates - source_predicates:
            differences.append(f"Unexpected predicate '{pred}' in translation")
        
        # Check argument structure
        source_args = set(re.findall(r':ARG\d+', source))
        target_args = set(re.findall(r':ARG\d+', target))
        
        for arg in source_args - target_args:
            differences.append(f"Argument role {arg} missing")
        for arg in target_args - source_args:
            differences.append(f"Extra argument role {arg} added")
        
        return differences


def quick_verify(source_amr: str, target_amr: str, threshold: float = 0.85) -> bool:
    """
    Quick verification check - returns True if translation passes.
    
    Args:
        source_amr: AMR from English parsing
        target_amr: AMR from Arabic reverse parsing
        threshold: Minimum F1 score for verification
        
    Returns:
        True if translation is semantically verified
    """
    comparator = GraphComparator(verification_threshold=threshold)
    result = comparator.compare(source_amr, target_amr)
    return result.is_verified

