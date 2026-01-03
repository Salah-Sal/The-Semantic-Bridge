"""
Arabic Generator - The "Renderer"

Takes AMR graphs and renders them into natural Arabic text
using an LLM constrained by the semantic structure.
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from .prompts import ARABIC_GENERATION_SYSTEM, ARABIC_GENERATION_USER

logger = logging.getLogger(__name__)


@dataclass
class ArabicGenerationResult:
    """Result of Arabic generation from AMR."""
    source_amr: str
    arabic_text: str
    transliteration: str
    semantic_notes: str
    success: bool = True
    error: Optional[str] = None


class ArabicGenerator:
    """
    Cross-Lingual Generator that renders AMR graphs into Arabic.
    
    Uses an LLM with carefully crafted prompts to ensure semantic
    fidelity - the Arabic output must express EXACTLY what is in
    the AMR graph, no more, no less.
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize the Arabic generator.
        
        Args:
            provider: LLM provider ("openai" or "anthropic")
            model: Model name (defaults to gpt-4-turbo or claude-3-opus)
            api_key: API key (defaults to environment variable)
        """
        self.provider = provider
        
        if provider == "openai":
            self.model = model or "gpt-4-turbo-preview"
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self._client = None
        elif provider == "anthropic":
            self.model = model or "claude-3-opus-20240229"
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            self._client = None
        elif provider == "gemini":
            self.model = model or "gemini-2.5-flash"
            self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            self._client = None
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai', 'anthropic', or 'gemini'")
    
    def _ensure_client(self):
        """Lazy initialization of the LLM client."""
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
    
    def generate(
        self,
        amr_graph: str,
        english_text: str = "",
        temperature: float = 0.3
    ) -> ArabicGenerationResult:
        """
        Generate Arabic text from an AMR graph.
        
        Args:
            amr_graph: AMR graph in PENMAN notation
            english_text: Original English (for context only)
            temperature: LLM temperature (lower = more deterministic)
            
        Returns:
            ArabicGenerationResult with Arabic text and metadata
        """
        self._ensure_client()
        
        user_prompt = ARABIC_GENERATION_USER.format(
            amr_graph=amr_graph,
            english_text=english_text or "(not provided)"
        )
        
        try:
            if self.provider == "openai":
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": ARABIC_GENERATION_SYSTEM},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=15000
                )
                content = response.choices[0].message.content
                
            elif self.provider == "anthropic":
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=15000,
                    system=ARABIC_GENERATION_SYSTEM,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )
                content = response.content[0].text
            
            elif self.provider == "gemini":
                # Gemini uses a combined prompt approach
                from google.genai import types
                full_prompt = f"{ARABIC_GENERATION_SYSTEM}\n\n{user_prompt}"
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=8192  # Gemini limit
                    )
                )
                content = response.text
            
            # Parse the response to extract components
            return self._parse_response(content, amr_graph)
            
        except Exception as e:
            logger.error(f"Arabic generation failed: {e}")
            return ArabicGenerationResult(
                source_amr=amr_graph,
                arabic_text="",
                transliteration="",
                semantic_notes="",
                success=False,
                error=str(e)
            )
    
    def _parse_response(self, content: str, source_amr: str) -> ArabicGenerationResult:
        """Parse LLM response to extract Arabic text and metadata."""
        lines = content.strip().split("\n")
        
        arabic_text = ""
        transliteration = ""
        notes = []
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect section headers
            lower = line.lower()
            if lower.startswith("arabic") or lower.startswith("1."):
                current_section = "arabic"
                # Check if value is on same line
                if ":" in line:
                    arabic_text = line.split(":", 1)[1].strip()
                continue
            elif lower.startswith("transliteration") or lower.startswith("2."):
                current_section = "translit"
                if ":" in line:
                    transliteration = line.split(":", 1)[1].strip()
                continue
            elif lower.startswith("note") or lower.startswith("3.") or lower.startswith("semantic"):
                current_section = "notes"
                if ":" in line:
                    notes.append(line.split(":", 1)[1].strip())
                continue
            
            # Collect content based on current section
            if current_section == "arabic" and not arabic_text:
                # Look for Arabic text (contains Arabic characters)
                if any('\u0600' <= c <= '\u06FF' for c in line):
                    arabic_text = line
            elif current_section == "translit" and not transliteration:
                transliteration = line
            elif current_section == "notes":
                notes.append(line)
            elif not arabic_text and any('\u0600' <= c <= '\u06FF' for c in line):
                # Fallback: first line with Arabic characters
                arabic_text = line
        
        # Extract Arabic from parenthetical pattern if present
        if not arabic_text:
            # Try to find Arabic anywhere in content
            for line in lines:
                if any('\u0600' <= c <= '\u06FF' for c in line):
                    arabic_text = line
                    break
        
        # Clean up Arabic text (remove transliteration if inline)
        if "(" in arabic_text and ")" in arabic_text:
            # Extract just the Arabic part
            match = re.search(r'([^\(]+)\s*\(', arabic_text)
            if match:
                arabic_text = match.group(1).strip()
        
        return ArabicGenerationResult(
            source_amr=source_amr,
            arabic_text=arabic_text,
            transliteration=transliteration,
            semantic_notes="\n".join(notes),
            success=bool(arabic_text)
        )
    
    async def generate_async(
        self,
        amr_graph: str,
        english_text: str = "",
        temperature: float = 0.3
    ) -> ArabicGenerationResult:
        """Async version of generate for web applications."""
        # For now, wrap sync version - can be optimized later
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate(amr_graph, english_text, temperature)
        )


class MockArabicGenerator(ArabicGenerator):
    """
    Mock Arabic Generator for testing without API calls.
    """
    
    def __init__(self):
        """Initialize mock generator."""
        self.provider = "mock"
        self.model = "mock"
        self._client = None
    
    def _ensure_client(self):
        """No-op for mock."""
        pass
    
    def generate(
        self,
        amr_graph: str,
        english_text: str = "",
        temperature: float = 0.3
    ) -> ArabicGenerationResult:
        """Generate mock Arabic based on AMR patterns."""
        
        # Check for polarity (negation)
        has_negation = ":polarity -" in amr_graph
        
        # Simple pattern matching for demo
        if "approve-01" in amr_graph:
            if has_negation:
                return ArabicGenerationResult(
                    source_amr=amr_graph,
                    arabic_text="لم توافق اللجنة على القرار",
                    transliteration="lam tuwāfiq al-lajnatu ʿalā al-qarār",
                    semantic_notes="approve-01 → توافق, polarity - → لم, ARG0 → اللجنة, ARG1 → القرار",
                    success=True
                )
            else:
                return ArabicGenerationResult(
                    source_amr=amr_graph,
                    arabic_text="وافقت اللجنة على القرار",
                    transliteration="wāfaqat al-lajnatu ʿalā al-qarār",
                    semantic_notes="approve-01 → وافقت, ARG0 → اللجنة, ARG1 → القرار",
                    success=True
                )
        
        if "want-01" in amr_graph:
            return ArabicGenerationResult(
                source_amr=amr_graph,
                arabic_text="يريد الولد أن يذهب",
                transliteration="yurīdu al-waladu an yadhdhab",
                semantic_notes="want-01 → يريد, ARG0 → الولد, ARG1 go-02 → أن يذهب",
                success=True
            )
        
        if "believe-01" in amr_graph:
            if has_negation:
                return ArabicGenerationResult(
                    source_amr=amr_graph,
                    arabic_text="لا يؤمن بذلك",
                    transliteration="lā yu'min bi-dhālik",
                    semantic_notes="believe-01 → يؤمن, polarity - → لا",
                    success=True
                )
            return ArabicGenerationResult(
                source_amr=amr_graph,
                arabic_text="يؤمن بذلك",
                transliteration="yu'min bi-dhālik",
                semantic_notes="believe-01 → يؤمن",
                success=True
            )
        
        # Default fallback
        return ArabicGenerationResult(
            source_amr=amr_graph,
            arabic_text="النص المترجم",
            transliteration="al-naṣṣ al-mutarjam",
            semantic_notes="Generic translation",
            success=True
        )

