#!/usr/bin/env python3
"""
Run script for The Semantic Bridge.

Usage:
    python run.py              # Run web server
    python run.py --demo       # Run demo translation
    python run.py --test       # Run tests
"""

import argparse
import os
import sys

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_server():
    """Start the web server."""
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    print("\n" + "=" * 60)
    print("  THE SEMANTIC BRIDGE | جسر المعنى")
    print("  English → Arabic Semantic Translation")
    print("=" * 60)
    print(f"\n  Starting server at http://{host}:{port}")
    print("  Press Ctrl+C to stop\n")
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True
    )


def run_demo():
    """Run a demo translation."""
    from src.pipeline import translate
    
    sentences = [
        "The committee did not approve the decision.",
        "The boy wants to go to school.",
        "She believes that he is honest.",
    ]
    
    print("\n" + "=" * 60)
    print("  THE SEMANTIC BRIDGE - Demo Mode")
    print("=" * 60)
    
    for sentence in sentences:
        print(f"\n{'─' * 60}")
        print(f"English: {sentence}")
        print("─" * 60)
        
        result = translate(sentence, use_mock=True)
        
        print(f"\nSource AMR:")
        for line in result.source_amr.split('\n'):
            print(f"  {line}")
        
        print(f"\nArabic: {result.arabic_text}")
        print(f"Transliteration: {result.transliteration}")
        
        print(f"\nReconstructed AMR:")
        for line in result.reconstructed_amr.split('\n'):
            print(f"  {line}")
        
        status = "✓ VERIFIED" if result.is_verified else "✗ FAILED"
        print(f"\nVerification: {status}")
        print(f"Smatch Score: {result.smatch_score:.1%}")
        
        if result.differences:
            print(f"Differences: {', '.join(result.differences)}")
    
    print("\n" + "=" * 60 + "\n")


def run_tests():
    """Run the test suite."""
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])


def main():
    parser = argparse.ArgumentParser(
        description="The Semantic Bridge - English to Arabic Semantic Translation"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo translations"
    )
    parser.add_argument(
        "--test",
        action="store_true", 
        help="Run test suite"
    )
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
    elif args.test:
        run_tests()
    else:
        run_server()


if __name__ == "__main__":
    main()

