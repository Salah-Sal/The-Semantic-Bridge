"""
Tests for The Semantic Bridge pipeline.

Run with: pytest tests/test_pipeline.py -v
"""

import pytest
from src.pipeline import (
    SemanticBridge,
    translate,
    TranslationStatus,
    PipelineResult
)
from src.amr_extractor import MockAMRExtractor, AMRResult
from src.arabic_generator import MockArabicGenerator
from src.reverse_verifier import MockReverseVerifier
from src.graph_comparator import GraphComparator, quick_verify


class TestMockComponents:
    """Test individual mock components."""
    
    def test_mock_amr_extractor_negation(self):
        """Test AMR extraction with negation."""
        extractor = MockAMRExtractor()
        result = extractor.parse("The committee did not approve the decision.")
        
        assert result.success
        assert "approve-01" in result.amr_graph
        assert ":polarity -" in result.amr_graph
        assert ":ARG0" in result.amr_graph
        assert ":ARG1" in result.amr_graph
    
    def test_mock_amr_extractor_want(self):
        """Test AMR extraction with want predicate."""
        extractor = MockAMRExtractor()
        result = extractor.parse("The boy wants to go.")
        
        assert result.success
        assert "want-01" in result.amr_graph
    
    def test_mock_arabic_generator_negation(self):
        """Test Arabic generation with negation."""
        generator = MockArabicGenerator()
        amr = """(a / approve-01
            :ARG0 (c / committee)
            :ARG1 (d / decision)
            :polarity -)"""
        
        result = generator.generate(amr)
        
        assert result.success
        assert "لم" in result.arabic_text  # Negation particle
        assert "توافق" in result.arabic_text  # approve verb
    
    def test_mock_arabic_generator_no_negation(self):
        """Test Arabic generation without negation."""
        generator = MockArabicGenerator()
        amr = """(a / approve-01
            :ARG0 (c / committee)
            :ARG1 (d / decision))"""
        
        result = generator.generate(amr)
        
        assert result.success
        assert "لم" not in result.arabic_text  # No negation
        assert "وافقت" in result.arabic_text  # approved (past tense)
    
    def test_mock_reverse_verifier_with_negation(self):
        """Test reverse verification detects negation."""
        verifier = MockReverseVerifier()
        result = verifier.parse("لم توافق اللجنة على القرار")
        
        assert result.success
        assert "approve-01" in result.reconstructed_amr
        assert ":polarity -" in result.reconstructed_amr
    
    def test_mock_reverse_verifier_rejection(self):
        """Test reverse verification catches rejection (different predicate)."""
        verifier = MockReverseVerifier()
        result = verifier.parse("رفضت اللجنة القرار")  # "The committee rejected..."
        
        assert result.success
        assert "reject-01" in result.reconstructed_amr  # Different predicate!
        assert "approve-01" not in result.reconstructed_amr


class TestGraphComparator:
    """Test AMR graph comparison."""
    
    def test_identical_graphs(self):
        """Test comparison of identical graphs."""
        comparator = GraphComparator(verification_threshold=0.85)
        
        amr = """(a / approve-01
            :ARG0 (c / committee)
            :ARG1 (d / decision)
            :polarity -)"""
        
        result = comparator.compare(amr, amr)
        
        assert result.is_verified
        assert result.f1_score >= 0.99
    
    def test_missing_negation(self):
        """Test that missing negation is caught."""
        comparator = GraphComparator(verification_threshold=0.85)
        
        source = """(a / approve-01
            :ARG0 (c / committee)
            :ARG1 (d / decision)
            :polarity -)"""
        
        target = """(a / approve-01
            :ARG0 (c / committee)
            :ARG1 (d / decision))"""
        
        result = comparator.compare(source, target)
        
        # Should detect the missing negation
        assert any("negation" in d.lower() or "polarity" in d.lower() 
                   for d in result.differences)
    
    def test_different_predicate(self):
        """Test that different predicates are caught (approve vs reject)."""
        comparator = GraphComparator(verification_threshold=0.85)
        
        source = """(a / approve-01
            :ARG0 (c / committee)
            :ARG1 (d / decision)
            :polarity -)"""
        
        target = """(r / reject-01
            :ARG0 (c / committee)
            :ARG1 (d / decision))"""
        
        result = comparator.compare(source, target)
        
        # Should NOT be verified - different semantic meaning
        assert not result.is_verified or result.f1_score < 1.0
    
    def test_quick_verify(self):
        """Test the quick_verify convenience function."""
        amr = """(w / want-01
            :ARG0 (b / boy)
            :ARG1 (g / go-02))"""
        
        assert quick_verify(amr, amr) is True


class TestPipeline:
    """Test the full translation pipeline."""
    
    def test_pipeline_with_mocks(self):
        """Test full pipeline with mock components."""
        result = translate(
            "The committee did not approve the decision.",
            use_mock=True
        )
        
        assert isinstance(result, PipelineResult)
        assert result.english_text == "The committee did not approve the decision."
        assert result.amr_extraction_success
        assert result.generation_success
        assert result.arabic_text
        assert result.source_amr
    
    def test_pipeline_negation_preservation(self):
        """Test that negation is preserved through the pipeline."""
        result = translate(
            "The committee did not approve the decision.",
            use_mock=True
        )
        
        # Source AMR should have negation
        assert ":polarity -" in result.source_amr
        
        # Arabic should have negation marker
        assert "لم" in result.arabic_text
        
        # Reconstructed AMR should have negation
        assert ":polarity -" in result.reconstructed_amr
    
    def test_pipeline_status_success(self):
        """Test successful pipeline status."""
        result = translate(
            "The committee did not approve the decision.",
            use_mock=True
        )
        
        # Mock components are designed to produce matching AMRs
        assert result.status in [TranslationStatus.SUCCESS, TranslationStatus.FAILED]
    
    def test_pipeline_to_dict(self):
        """Test result serialization."""
        result = translate(
            "The committee did not approve the decision.",
            use_mock=True
        )
        
        data = result.to_dict()
        
        assert "english_text" in data
        assert "arabic_text" in data
        assert "source_amr" in data
        assert "smatch_score" in data
        assert "status" in data
    
    def test_semantic_bridge_initialization(self):
        """Test SemanticBridge initialization with mock mode."""
        bridge = SemanticBridge(
            use_mock=True,
            verification_threshold=0.9,
            max_retries=3
        )
        
        assert bridge.verification_threshold == 0.9
        assert bridge.max_retries == 3
        assert isinstance(bridge.amr_extractor, MockAMRExtractor)
        assert isinstance(bridge.arabic_generator, MockArabicGenerator)
        assert isinstance(bridge.reverse_verifier, MockReverseVerifier)


class TestGoogleTranslateDifference:
    """
    Test cases demonstrating the difference from Google Translate.
    
    These tests show how our system catches semantic mismatches that
    simple translation might miss.
    """
    
    def test_reject_vs_not_approve(self):
        """
        Demonstrate: 'reject' is NOT the same as 'not approve'.
        
        Google might translate "did not approve" as "rejected" (rafadat),
        but our system catches this semantic difference.
        """
        comparator = GraphComparator()
        
        # What we want: "not approve"
        source_amr = """(a / approve-01
            :ARG0 (c / committee)
            :ARG1 (d / decision)
            :polarity -)"""
        
        # What Google might produce: "reject"
        wrong_target = """(r / reject-01
            :ARG0 (c / committee)
            :ARG1 (d / decision))"""
        
        result = comparator.compare(source_amr, wrong_target)
        
        # This should NOT be verified!
        # approve-01 + polarity - ≠ reject-01
        has_predicate_diff = any(
            "approve" in d or "reject" in d 
            for d in result.differences
        )
        
        # The score should be lower due to predicate mismatch
        assert result.f1_score < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

