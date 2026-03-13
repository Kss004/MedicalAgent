"""
Verification script for Knowledge Base retrieval context.
"""
import sys
import os
import asyncio

# Add project paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medical_assistant import search_trusted_sources

async def verify():
    test_queries = [
        "What are the metabolic effects of PCOS?",
        "What is the recommended diet for someone with PCOS?",
        "How does PCOS affect insulin resistance?"
    ]
    
    print("=== Retrieval Verification ===")
    for q in test_queries:
        print(f"\nQuery: {q}")
        results = search_trusted_sources(q)
        context = results.get("context", "")
        sources = results.get("source_urls", [])
        
        print(f"Sources found: {len(sources)}")
        print(f"Context length: {len(context)} characters")
        if len(context) > 200:
            print(f"Sample Content: {context[:200]}...")
        else:
            print("WARNING: Context is very short!")

if __name__ == "__main__":
    asyncio.run(verify())
