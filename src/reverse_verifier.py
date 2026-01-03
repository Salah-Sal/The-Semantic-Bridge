"""
Reverse Verifier - The "Arabic Auditor"

Parses Arabic text back into AMR using LLM-based zero-shot parsing.
This enables verification by comparing the reconstructed graph with the source.
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional
import penman

from .prompts import REVERSE_PARSING_SYSTEM, REVERSE_PARSING_USER

logger = logging.getLogger(__name__)


@dataclass
class ReverseParseResult:
    """Result of reverse-parsing Arabic to AMR."""
    arabic_text: str
    reconstructed_amr: str
    penman_graph: Optional[penman.Graph] = None
    success: bool = True
    error: Optional[str] = None


class ReverseVerifier:
    """
    LLM-based Zero-Shot Arabic to AMR Parser.
    
    Since there are no robust open-source Arabic AMR parsers,
    we use an LLM to perform cross-lingual parsing. The LLM
    understands that Arabic words map to English PropBank predicates
    (e.g., يؤمن → believe-01).
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize the reverse verifier.
        
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
    
    def parse(
        self,
        arabic_text: str,
        transliteration: str = "",
        temperature: float = 0.1  # Low temp for consistent parsing
    ) -> ReverseParseResult:
        """
        Parse Arabic text into an AMR graph.
        
        Args:
            arabic_text: Arabic sentence to parse
            transliteration: Optional transliteration to help the LLM
            temperature: LLM temperature (lower = more deterministic)
            
        Returns:
            ReverseParseResult with reconstructed AMR graph
        """
        self._ensure_client()
        
        user_prompt = REVERSE_PARSING_USER.format(
            arabic_text=arabic_text,
            transliteration=transliteration or "(not provided)"
        )
        
        try:
            if self.provider == "openai":
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": REVERSE_PARSING_SYSTEM},
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
                    system=REVERSE_PARSING_SYSTEM,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )
                content = response.content[0].text
            
            elif self.provider == "gemini":
                # Gemini uses a combined prompt approach
                from google.genai import types
                full_prompt = f"{REVERSE_PARSING_SYSTEM}\n\n{user_prompt}"
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=8192  # Gemini limit
                    )
                )
                content = response.text
            
            # Extract and validate AMR from response
            return self._parse_response(content, arabic_text)
            
        except Exception as e:
            logger.error(f"Reverse parsing failed: {e}")
            return ReverseParseResult(
                arabic_text=arabic_text,
                reconstructed_amr="",
                success=False,
                error=str(e)
            )
    
    def _parse_response(self, content: str, arabic_text: str) -> ReverseParseResult:
        """Extract and validate AMR from LLM response."""
        
        # Try to extract AMR graph from response
        amr_string = self._extract_amr(content)
        
        if not amr_string:
            return ReverseParseResult(
                arabic_text=arabic_text,
                reconstructed_amr="",
                success=False,
                error="Could not extract AMR from response"
            )
        
        # Validate by parsing with penman
        try:
            penman_graph = penman.decode(amr_string)
            # Re-encode for consistent formatting
            amr_string = penman.encode(penman_graph, indent=2)
        except Exception as e:
            logger.warning(f"Could not parse AMR with penman: {e}")
            penman_graph = None
        
        return ReverseParseResult(
            arabic_text=arabic_text,
            reconstructed_amr=amr_string,
            penman_graph=penman_graph,
            success=True
        )
    
    def _extract_amr(self, content: str) -> str:
        """Extract AMR graph from LLM response."""
        
        # Look for code block
        code_block_match = re.search(r'```(?:amr|penman)?\s*([\s\S]*?)```', content)
        if code_block_match:
            return code_block_match.group(1).strip()
        
        # Look for parenthetical AMR structure
        # AMR graphs start with ( and a variable like (a / approve-01
        amr_match = re.search(r'(\([a-z]\d?\s*/\s*\S+[\s\S]*)', content)
        if amr_match:
            amr_text = amr_match.group(1)
            # Try to find matching closing paren
            depth = 0
            end_idx = 0
            for i, c in enumerate(amr_text):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                    if depth == 0:
                        end_idx = i + 1
                        break
            if end_idx > 0:
                return amr_text[:end_idx]
        
        # Fallback: entire content if it looks like AMR
        content = content.strip()
        if content.startswith("(") and "/" in content:
            return content
        
        return ""
    
    async def parse_async(
        self,
        arabic_text: str,
        transliteration: str = "",
        temperature: float = 0.1
    ) -> ReverseParseResult:
        """Async version of parse for web applications."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.parse(arabic_text, transliteration, temperature)
        )


class MockReverseVerifier(ReverseVerifier):
    """
    Mock Reverse Verifier for testing without API calls.
    """
    
    def __init__(self):
        """Initialize mock verifier."""
        self.provider = "mock"
        self.model = "mock"
        self._client = None
    
    def _ensure_client(self):
        """No-op for mock."""
        pass
    
    def parse(
        self,
        arabic_text: str,
        transliteration: str = "",
        temperature: float = 0.1
    ) -> ReverseParseResult:
        """Generate mock AMR based on Arabic patterns."""
        
        # Check for negation markers
        has_negation = any(neg in arabic_text for neg in ["لم", "لا", "ما", "ليس", "لن"])
        
        # Pattern matching based on Arabic text
        if "توافق" in arabic_text or "وافق" in arabic_text:
            amr = """(a / approve-01
    :ARG0 (c / committee)
    :ARG1 (d / decision)"""
            if has_negation:
                amr += "\n    :polarity -)"
            else:
                amr += ")"
                
            return ReverseParseResult(
                arabic_text=arabic_text,
                reconstructed_amr=amr,
                success=True
            )
        
        if "يريد" in arabic_text:
            return ReverseParseResult(
                arabic_text=arabic_text,
                reconstructed_amr="""(w / want-01
    :ARG0 (b / boy)
    :ARG1 (g / go-02
        :ARG0 b))""",
                success=True
            )
        
        if "يؤمن" in arabic_text or "يعتقد" in arabic_text:
            amr = """(b / believe-01
    :ARG0 (p / person)
    :ARG1 (t / thing)"""
            if has_negation:
                amr += "\n    :polarity -)"
            else:
                amr += ")"
                
            return ReverseParseResult(
                arabic_text=arabic_text,
                reconstructed_amr=amr,
                success=True
            )
        
        # Check for rejection (to test the Google Translate difference)
        if "رفض" in arabic_text:
            # This would NOT match approve-01 !
            return ReverseParseResult(
                arabic_text=arabic_text,
                reconstructed_amr="""(r / reject-01
    :ARG0 (c / committee)
    :ARG1 (d / decision))""",
                success=True
            )
        
        # Default fallback
        return ReverseParseResult(
            arabic_text=arabic_text,
            reconstructed_amr="""(s / say-01
    :ARG0 (p / person)
    :ARG1 (t / thing))""",
            success=True
        )

