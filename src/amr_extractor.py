"""
AMR Extractor - The "Source of Truth"

Parses English text into Abstract Meaning Representation (AMR) graphs
using the amrlib library (~83%+ F1 score).
"""

import logging
from dataclasses import dataclass
from typing import Optional
import penman

logger = logging.getLogger(__name__)


@dataclass
class AMRResult:
    """Result of AMR parsing."""
    original_text: str
    amr_graph: str
    penman_graph: Optional[penman.Graph] = None
    success: bool = True
    error: Optional[str] = None


class AMRExtractor:
    """
    English AMR Parser using amrlib.
    
    This is the "Source of Truth" - the authoritative semantic
    representation that all translations must preserve.
    """
    
    def __init__(self, model_name: str = "parse_xfm_bart_large"):
        """
        Initialize the AMR extractor.
        
        Args:
            model_name: The amrlib model to use. Options:
                - parse_xfm_bart_large (recommended, highest accuracy)
                - parse_xfm_bart_base (faster, slightly lower accuracy)
                - parse_spring (alternative architecture)
        """
        self.model_name = model_name
        self._parser = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """Lazy initialization of the AMR parser."""
        if self._initialized:
            return
        
        try:
            import amrlib
            logger.info(f"Loading AMR parser model: {self.model_name}")
            
            # Load the parsing model
            self._parser = amrlib.load_stog_model()
            self._initialized = True
            logger.info("AMR parser initialized successfully")
            
        except ImportError as e:
            logger.error("amrlib not installed. Install with: pip install amrlib")
            raise ImportError(
                "amrlib is required for AMR parsing. "
                "Install with: pip install amrlib\n"
                "Then download models with: python -c \"import amrlib; amrlib.download_model('parse_xfm_bart_large')\""
            ) from e
        except Exception as e:
            logger.error(f"Failed to initialize AMR parser: {e}")
            raise
    
    def parse(self, text: str) -> AMRResult:
        """
        Parse English text into AMR.
        
        Args:
            text: English sentence to parse
            
        Returns:
            AMRResult containing the AMR graph in PENMAN notation
        """
        self._ensure_initialized()
        
        try:
            # Parse the sentence
            graphs = self._parser.parse_sents([text])
            
            if not graphs or graphs[0] is None:
                return AMRResult(
                    original_text=text,
                    amr_graph="",
                    success=False,
                    error="Parser returned no result"
                )
            
            amr_string = graphs[0]
            
            # Parse into penman graph for validation and manipulation
            try:
                penman_graph = penman.decode(amr_string)
            except Exception as e:
                logger.warning(f"Could not decode AMR as penman graph: {e}")
                penman_graph = None
            
            return AMRResult(
                original_text=text,
                amr_graph=amr_string,
                penman_graph=penman_graph,
                success=True
            )
            
        except Exception as e:
            logger.error(f"AMR parsing failed: {e}")
            return AMRResult(
                original_text=text,
                amr_graph="",
                success=False,
                error=str(e)
            )
    
    def parse_batch(self, texts: list[str]) -> list[AMRResult]:
        """
        Parse multiple sentences into AMR graphs.
        
        Args:
            texts: List of English sentences
            
        Returns:
            List of AMRResult objects
        """
        self._ensure_initialized()
        
        try:
            graphs = self._parser.parse_sents(texts)
            results = []
            
            for text, graph in zip(texts, graphs):
                if graph is None:
                    results.append(AMRResult(
                        original_text=text,
                        amr_graph="",
                        success=False,
                        error="Parser returned no result"
                    ))
                else:
                    try:
                        penman_graph = penman.decode(graph)
                    except Exception:
                        penman_graph = None
                    
                    results.append(AMRResult(
                        original_text=text,
                        amr_graph=graph,
                        penman_graph=penman_graph,
                        success=True
                    ))
            
            return results
            
        except Exception as e:
            logger.error(f"Batch AMR parsing failed: {e}")
            return [
                AMRResult(
                    original_text=text,
                    amr_graph="",
                    success=False,
                    error=str(e)
                )
                for text in texts
            ]
    
    @staticmethod
    def normalize_amr(amr_string: str) -> str:
        """
        Normalize an AMR string for consistent comparison.
        
        Args:
            amr_string: AMR in PENMAN notation
            
        Returns:
            Normalized AMR string
        """
        try:
            graph = penman.decode(amr_string)
            # Re-encode with consistent formatting
            return penman.encode(graph, indent=2)
        except Exception:
            return amr_string


class LLMAMRExtractor(AMRExtractor):
    """
    LLM-based AMR Extractor for when amrlib is not available.
    
    Uses an LLM to generate AMR graphs from English text.
    """
    
    AMR_PARSING_PROMPT = """You are an expert in Abstract Meaning Representation (AMR) parsing.

Convert the following English sentence into a PropBank-based AMR graph.

## AMR Guidelines:
1. Use PropBank frame IDs for predicates (e.g., want-01, believe-01, approve-01, be-located-at-91)
2. Use standard argument roles: :ARG0 (agent), :ARG1 (patient/theme), :ARG2 (instrument/beneficiary)
3. Use lowercase single-letter variables (a, b, c, s, t, etc.) - each variable must be unique
4. Mark negation with :polarity -
5. Use proper PENMAN notation with indentation
6. For complex sentences with multiple clauses, use :conj for coordination
7. For technical terms, preserve them as concept names (e.g., supertagging → supertag-01 or supertag)
8. Use :mod for adjectives/modifiers, :manner for adverbs
9. Use :purpose for "in order to", :condition for "if"

## Complex Sentence Example:
Sentence: "Machine learning is essential for natural language processing and is widely used in industry."
AMR:
(a / and
    :op1 (e / essential-01
        :ARG1 (l / learn-01
            :ARG1 (m / machine))
        :ARG2 (p / process-01
            :ARG1 (l2 / language
                :mod (n / natural))))
    :op2 (u / use-01
        :ARG1 l
        :manner (w / wide)
        :location (i / industry)))

## Your Task:
Parse this sentence into AMR (handle all clauses, preserve technical terms):
"{text}"

Output ONLY the AMR graph in PENMAN notation, no explanations."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = None,
        api_key: str = None
    ):
        """Initialize LLM-based extractor."""
        import os
        self.provider = provider
        
        # Set default model based on provider
        if model:
            self.model = model
        elif provider == "openai":
            self.model = "gpt-4-turbo-preview"
        elif provider == "anthropic":
            self.model = "claude-3-opus-20240229"
        elif provider == "gemini":
            self.model = "gemini-2.5-flash"
        else:
            self.model = "gpt-4-turbo-preview"
        
        # Get API key based on provider
        if api_key:
            self.api_key = api_key
        elif provider == "gemini":
            self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        elif provider == "anthropic":
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")
        
        self._client = None
        self._initialized = True
        self._parser = None
    
    def _ensure_client(self):
        """Initialize the LLM client."""
        if self._client is not None:
            return
        
        if self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        elif self.provider == "anthropic":
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self.api_key)
        elif self.provider == "gemini":
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
    
    def _ensure_initialized(self):
        """No-op for LLM extractor."""
        pass
    
    def parse(self, text: str) -> AMRResult:
        """Parse English text into AMR using an LLM."""
        self._ensure_client()
        
        prompt = self.AMR_PARSING_PROMPT.format(text=text)
        
        try:
            if self.provider == "openai":
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=15000
                )
                content = response.choices[0].message.content
            elif self.provider == "anthropic":
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=15000,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text
            
            elif self.provider == "gemini":
                from google.genai import types
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=8192  # Gemini limit
                    )
                )
                content = response.text
            
            # Extract AMR from response
            amr_graph = self._extract_amr(content)
            
            # Validate with penman
            try:
                penman_graph = penman.decode(amr_graph)
                amr_graph = penman.encode(penman_graph, indent=4)
            except Exception:
                pass
            
            return AMRResult(
                original_text=text,
                amr_graph=amr_graph,
                success=True
            )
            
        except Exception as e:
            logger.error(f"LLM AMR parsing failed: {e}")
            return AMRResult(
                original_text=text,
                amr_graph="",
                success=False,
                error=str(e)
            )
    
    def _extract_amr(self, content: str) -> str:
        """Extract AMR graph from LLM response."""
        import re
        
        # Look for code block
        code_match = re.search(r'```(?:amr|penman)?\s*([\s\S]*?)```', content)
        if code_match:
            return code_match.group(1).strip()
        
        # Look for AMR structure starting with (
        amr_match = re.search(r'(\([a-z]\d?\s*/\s*\S+[\s\S]*)', content)
        if amr_match:
            amr_text = amr_match.group(1)
            # Find matching closing paren
            depth = 0
            for i, c in enumerate(amr_text):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                    if depth == 0:
                        return amr_text[:i+1]
        
        return content.strip()


class MockAMRExtractor(AMRExtractor):
    """
    Mock AMR Extractor for testing without loading heavy models.
    
    Uses pattern matching for known test cases, falls back to LLM if available.
    """
    
    def __init__(self, llm_client=None, use_llm_fallback: bool = True):
        """Initialize mock extractor with optional LLM fallback."""
        self.llm_client = llm_client
        self.use_llm_fallback = use_llm_fallback
        self._llm_extractor = None
        self._initialized = True
        self._parser = None
    
    def _ensure_initialized(self):
        """No-op for mock."""
        pass
    
    def _get_llm_extractor(self):
        """Lazy init LLM extractor for fallback."""
        if self._llm_extractor is None and self.use_llm_fallback:
            import os
            if os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
                try:
                    provider = "openai" if os.getenv("OPENAI_API_KEY") else "anthropic"
                    self._llm_extractor = LLMAMRExtractor(provider=provider)
                except Exception as e:
                    logger.warning(f"Could not init LLM extractor: {e}")
        return self._llm_extractor
    
    def parse(self, text: str) -> AMRResult:
        """
        Generate AMR using pattern matching or LLM fallback.
        
        For MVP testing without loading the full amrlib model.
        """
        # Simple pattern-based mock for common test cases
        text_lower = text.lower()
        
        # Pattern: "X did not approve Y"
        if "did not approve" in text_lower or "didn't approve" in text_lower:
            return AMRResult(
                original_text=text,
                amr_graph="""(a / approve-01
    :ARG0 (c / committee)
    :ARG1 (d / decision)
    :polarity -)""",
                success=True
            )
        
        # Pattern: "X wants to Y"
        if "want" in text_lower:
            return AMRResult(
                original_text=text,
                amr_graph="""(w / want-01
    :ARG0 (p / person)
    :ARG1 (g / go-02
        :ARG0 p))""",
                success=True
            )
        
        # Pattern: "X believes Y"
        if "believe" in text_lower:
            return AMRResult(
                original_text=text,
                amr_graph="""(b / believe-01
    :ARG0 (p / person)
    :ARG1 (t / true))""",
                success=True
            )
        
        # Try LLM fallback for unknown patterns
        llm_extractor = self._get_llm_extractor()
        if llm_extractor:
            logger.info("Using LLM fallback for AMR parsing")
            return llm_extractor.parse(text)
        
        # Default fallback (simple generic graph)
        return AMRResult(
            original_text=text,
            amr_graph="""(s / say-01
    :ARG0 (p / person)
    :ARG1 (t / thing))""",
            success=True
        )

