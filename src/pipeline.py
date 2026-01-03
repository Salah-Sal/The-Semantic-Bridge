"""
The Semantic Bridge - Main Translation Pipeline

Orchestrates the full semantic translation workflow:
1. English → AMR (Source of Truth)
2. AMR → Arabic (Rendering)
3. Arabic → AMR (Verification)
4. AMR ↔ AMR (Comparison)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from .amr_extractor import AMRExtractor, AMRResult, MockAMRExtractor, LLMAMRExtractor
from .arabic_generator import ArabicGenerator, ArabicGenerationResult, MockArabicGenerator
from .reverse_verifier import ReverseVerifier, ReverseParseResult, MockReverseVerifier
from .graph_comparator import GraphComparator, ComparisonResult

logger = logging.getLogger(__name__)


class TranslationStatus(str, Enum):
    """Status of the translation pipeline."""
    SUCCESS = "success"           # Verified translation
    RETRY = "retry"               # Failed verification, retrying
    FAILED = "failed"             # Failed after max retries
    PARTIAL = "partial"           # Some steps succeeded
    ERROR = "error"               # System error


@dataclass
class PipelineResult:
    """Complete result of the semantic translation pipeline."""
    # Input
    english_text: str
    
    # Step 1: AMR Extraction
    source_amr: str = ""
    amr_extraction_success: bool = False
    
    # Step 2: Arabic Generation
    arabic_text: str = ""
    transliteration: str = ""
    semantic_notes: str = ""
    generation_success: bool = False
    
    # Step 3: Reverse Verification
    reconstructed_amr: str = ""
    verification_parse_success: bool = False
    
    # Step 4: Comparison
    smatch_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    is_verified: bool = False
    differences: list[str] = field(default_factory=list)
    
    # Overall Status
    status: TranslationStatus = TranslationStatus.ERROR
    attempts: int = 0
    error_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization with step-by-step details."""
        return {
            "english_text": self.english_text,
            "source_amr": self.source_amr,
            "arabic_text": self.arabic_text,
            "transliteration": self.transliteration,
            "semantic_notes": self.semantic_notes,
            "reconstructed_amr": self.reconstructed_amr,
            "smatch_score": round(self.smatch_score, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "is_verified": self.is_verified,
            "differences": self.differences,
            "status": self.status.value,
            "attempts": self.attempts,
            "error_message": self.error_message,
            # Step-by-step pipeline data
            "steps": [
                {
                    "number": 1,
                    "name": "AMR Extraction",
                    "description": "Parse English into Abstract Meaning Representation",
                    "status": "success" if self.amr_extraction_success else ("error" if self.error_message and "AMR extraction" in self.error_message else "pending"),
                    "input": {
                        "label": "English Text",
                        "value": self.english_text,
                        "type": "english"
                    },
                    "output": {
                        "label": "AMR Graph (PropBank)",
                        "value": self.source_amr,
                        "type": "amr"
                    },
                    "details": "Using amrlib parser (~83% F1 accuracy)"
                },
                {
                    "number": 2,
                    "name": "Arabic Generation",
                    "description": "Render AMR graph into natural Arabic",
                    "status": "success" if self.generation_success else ("error" if self.error_message and "Arabic generation" in self.error_message else "pending"),
                    "input": {
                        "label": "AMR Graph",
                        "value": self.source_amr,
                        "type": "amr"
                    },
                    "output": {
                        "label": "Arabic Translation",
                        "value": self.arabic_text,
                        "type": "arabic",
                        "transliteration": self.transliteration,
                        "notes": self.semantic_notes
                    },
                    "details": "LLM constrained to express only AMR content"
                },
                {
                    "number": 3,
                    "name": "Reverse Parsing",
                    "description": "Parse Arabic back into AMR for verification",
                    "status": "success" if self.verification_parse_success else ("error" if self.error_message and "Reverse parsing" in self.error_message else "pending"),
                    "input": {
                        "label": "Arabic Text",
                        "value": self.arabic_text,
                        "type": "arabic"
                    },
                    "output": {
                        "label": "Reconstructed AMR",
                        "value": self.reconstructed_amr,
                        "type": "amr"
                    },
                    "details": "Zero-shot cross-lingual AMR parsing via LLM"
                },
                {
                    "number": 4,
                    "name": "Semantic Comparison",
                    "description": "Compare source and reconstructed AMR graphs",
                    "status": "success" if self.is_verified else ("error" if self.smatch_score > 0 else "pending"),
                    "input": {
                        "label": "Source vs Reconstructed AMR",
                        "value": {
                            "source": self.source_amr,
                            "reconstructed": self.reconstructed_amr
                        },
                        "type": "comparison"
                    },
                    "output": {
                        "label": "Verification Result",
                        "value": {
                            "verified": self.is_verified,
                            "smatch_score": round(self.smatch_score, 4),
                            "precision": round(self.precision, 4),
                            "recall": round(self.recall, 4),
                            "differences": self.differences
                        },
                        "type": "verification"
                    },
                    "details": f"Smatch F1 threshold: 85% | Score: {round(self.smatch_score * 100, 1)}%"
                }
            ]
        }


class SemanticBridge:
    """
    The Semantic Bridge Translation Pipeline.
    
    Provides semantic-preserving translation from English to Arabic
    by extracting Universal Logic (AMR), rendering to Arabic, and
    verifying the semantic preservation.
    """
    
    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: Optional[str] = None,
        api_key: Optional[str] = None,
        verification_threshold: float = 0.85,
        max_retries: int = 2,
        use_mock: bool = False
    ):
        """
        Initialize the Semantic Bridge pipeline.
        
        Args:
            llm_provider: "openai" or "anthropic"
            llm_model: Model name for LLM components
            api_key: API key for LLM
            verification_threshold: Min Smatch score for verification
            max_retries: Maximum retry attempts on verification failure
            use_mock: Use mock components for testing
        """
        self.verification_threshold = verification_threshold
        self.max_retries = max_retries
        
        if use_mock:
            self.amr_extractor = MockAMRExtractor()
            self.arabic_generator = MockArabicGenerator()
            self.reverse_verifier = MockReverseVerifier()
        else:
            # Try to use amrlib if available, otherwise use LLM-based parsing
            try:
                import amrlib
                # Check if model is downloaded
                amrlib.load_stog_model()
                self.amr_extractor = AMRExtractor()
                logger.info("Using amrlib for AMR extraction")
            except Exception as e:
                logger.info(f"amrlib not available ({e}), using LLM-based AMR extraction")
                self.amr_extractor = LLMAMRExtractor(
                    provider=llm_provider,
                    model=llm_model,
                    api_key=api_key
                )
            
            self.arabic_generator = ArabicGenerator(
                provider=llm_provider,
                model=llm_model,
                api_key=api_key
            )
            self.reverse_verifier = ReverseVerifier(
                provider=llm_provider,
                model=llm_model,
                api_key=api_key
            )
        
        self.comparator = GraphComparator(
            verification_threshold=verification_threshold
        )
    
    def _estimate_complexity(self, text: str) -> float:
        """
        Estimate sentence complexity (0.0 = simple, 1.0 = very complex).
        Used to adjust verification threshold for complex sentences.
        """
        complexity = 0.0
        
        # Word count
        words = text.split()
        if len(words) > 20:
            complexity += 0.2
        if len(words) > 35:
            complexity += 0.2
        
        # Clause indicators
        clause_markers = ['and', 'but', 'which', 'that', 'because', 'although', 'while', 'if', 'when']
        for marker in clause_markers:
            if marker in text.lower():
                complexity += 0.1
        
        # Technical terms (heuristic)
        technical_indicators = ['parsing', 'grammar', 'algorithm', 'neural', 'semantic', 
                                'linguistic', 'computational', 'structure', 'analysis']
        for term in technical_indicators:
            if term in text.lower():
                complexity += 0.05
        
        return min(complexity, 1.0)
    
    def translate(
        self, 
        english_text: str, 
        adaptive_threshold: bool = True,
        provider_override: Optional[str] = None,
        model_override: Optional[str] = None
    ) -> PipelineResult:
        """
        Translate English text to Arabic with semantic verification.
        
        This is the main entry point that orchestrates the full pipeline.
        
        Args:
            english_text: English sentence to translate
            adaptive_threshold: Lower threshold for complex sentences
            provider_override: Override the default LLM provider
            model_override: Override the default model
            
        Returns:
            PipelineResult with translation and verification details
        """
        # If provider/model override requested, create temporary components
        if provider_override or model_override:
            self._apply_provider_override(provider_override, model_override)
        result = PipelineResult(english_text=english_text)
        
        # Adjust threshold for complex sentences
        if adaptive_threshold:
            complexity = self._estimate_complexity(english_text)
            if complexity > 0.3:
                # Lower threshold for complex sentences (0.85 → 0.60 for very complex)
                adjusted_threshold = self.verification_threshold - (complexity * 0.35)
                self.comparator.threshold = max(adjusted_threshold, 0.50)
                logger.info(f"Complex sentence detected (complexity={complexity:.2f}), "
                           f"adjusted threshold to {self.comparator.threshold:.2f}")
        
        # Step 1: Parse English to AMR
        logger.info(f"Step 1: Parsing English to AMR: '{english_text}'")
        amr_result = self._extract_amr(english_text)
        
        if not amr_result.success:
            result.status = TranslationStatus.ERROR
            result.error_message = f"AMR extraction failed: {amr_result.error}"
            return result
        
        result.source_amr = amr_result.amr_graph
        result.amr_extraction_success = True
        
        # Attempt translation with retries
        for attempt in range(1, self.max_retries + 2):
            result.attempts = attempt
            
            # Step 2: Generate Arabic from AMR
            logger.info(f"Step 2 (attempt {attempt}): Generating Arabic from AMR")
            gen_result = self._generate_arabic(
                amr_result.amr_graph,
                english_text,
                attempt
            )
            
            if not gen_result.success:
                result.status = TranslationStatus.ERROR
                result.error_message = f"Arabic generation failed: {gen_result.error}"
                continue
            
            result.arabic_text = gen_result.arabic_text
            result.transliteration = gen_result.transliteration
            result.semantic_notes = gen_result.semantic_notes
            result.generation_success = True
            
            # Step 3: Reverse parse Arabic to AMR
            logger.info(f"Step 3: Reverse parsing Arabic to AMR")
            verify_result = self._reverse_parse(
                gen_result.arabic_text,
                gen_result.transliteration
            )
            
            if not verify_result.success:
                result.status = TranslationStatus.PARTIAL
                result.error_message = f"Reverse parsing failed: {verify_result.error}"
                continue
            
            result.reconstructed_amr = verify_result.reconstructed_amr
            result.verification_parse_success = True
            
            # Step 4: Compare graphs
            logger.info(f"Step 4: Comparing source and reconstructed AMR")
            comp_result = self._compare_graphs(
                amr_result.amr_graph,
                verify_result.reconstructed_amr
            )
            
            result.smatch_score = comp_result.smatch_score
            result.precision = comp_result.precision
            result.recall = comp_result.recall
            result.is_verified = comp_result.is_verified
            result.differences = comp_result.differences
            
            if comp_result.is_verified:
                result.status = TranslationStatus.SUCCESS
                logger.info(f"Translation verified with Smatch F1={comp_result.f1_score:.3f}")
                return result
            else:
                logger.warning(
                    f"Verification failed (F1={comp_result.f1_score:.3f}), "
                    f"differences: {comp_result.differences}"
                )
                if attempt <= self.max_retries:
                    result.status = TranslationStatus.RETRY
                    continue
        
        # All retries exhausted
        result.status = TranslationStatus.FAILED
        result.error_message = (
            f"Translation could not be verified after {result.attempts} attempts. "
            f"Final Smatch score: {result.smatch_score:.3f}"
        )
        return result
    
    def _extract_amr(self, text: str) -> AMRResult:
        """Extract AMR from English text."""
        return self.amr_extractor.parse(text)
    
    def _generate_arabic(
        self,
        amr_graph: str,
        english_text: str,
        attempt: int
    ) -> ArabicGenerationResult:
        """Generate Arabic from AMR graph."""
        # Increase temperature slightly on retries for diversity
        temperature = 0.3 + (attempt - 1) * 0.1
        return self.arabic_generator.generate(
            amr_graph,
            english_text,
            temperature=min(temperature, 0.7)
        )
    
    def _reverse_parse(
        self,
        arabic_text: str,
        transliteration: str
    ) -> ReverseParseResult:
        """Reverse parse Arabic to AMR."""
        return self.reverse_verifier.parse(arabic_text, transliteration)
    
    def _compare_graphs(
        self,
        source_amr: str,
        target_amr: str
    ) -> ComparisonResult:
        """Compare source and target AMR graphs."""
        return self.comparator.compare(source_amr, target_amr)
    
    def _apply_provider_override(
        self, 
        provider: Optional[str] = None, 
        model: Optional[str] = None
    ):
        """
        Apply temporary provider/model override for this translation.
        
        Creates new generator and verifier instances with the specified provider.
        """
        import os
        
        provider = provider or self.llm_provider
        
        # Get API key for the provider
        if provider == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
        else:  # openai
            api_key = os.getenv("OPENAI_API_KEY")
        
        logger.info(f"Applying provider override: {provider}, model: {model or 'default'}")
        
        # Create new generator with override
        self.arabic_generator = ArabicGenerator(
            provider=provider,
            model=model,
            api_key=api_key
        )
        
        # Create new verifier with override
        self.reverse_verifier = ReverseVerifier(
            provider=provider,
            model=model,
            api_key=api_key
        )
        
        # If using LLM-based AMR extraction, update it too
        if isinstance(self.amr_extractor, LLMAMRExtractor):
            self.amr_extractor = LLMAMRExtractor(
                provider=provider,
                model=model,
                api_key=api_key
            )
    
    async def translate_async(
        self, 
        english_text: str,
        provider_override: Optional[str] = None,
        model_override: Optional[str] = None
    ) -> PipelineResult:
        """
        Async version of translate for web applications.
        
        For now wraps the sync version - can be fully async-ified later.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.translate(
                english_text,
                provider_override=provider_override,
                model_override=model_override
            )
        )


def create_pipeline(
    use_mock: bool = False,
    **kwargs
) -> SemanticBridge:
    """
    Factory function to create a configured pipeline.
    
    Args:
        use_mock: Use mock components for testing
        **kwargs: Additional arguments for SemanticBridge
        
    Returns:
        Configured SemanticBridge instance
    """
    return SemanticBridge(use_mock=use_mock, **kwargs)


# Convenience function for quick translation
def translate(
    english_text: str,
    use_mock: bool = False,
    **kwargs
) -> PipelineResult:
    """
    Quick translation function.
    
    Args:
        english_text: English text to translate
        use_mock: Use mock components for testing
        **kwargs: Additional pipeline configuration
        
    Returns:
        PipelineResult with translation
    """
    pipeline = create_pipeline(use_mock=use_mock, **kwargs)
    return pipeline.translate(english_text)

