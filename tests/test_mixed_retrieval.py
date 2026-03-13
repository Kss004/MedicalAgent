import sys
import os
import asyncio

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medical_assistant import ask_health_assistant

async def verify():
    query = "pcos symptoms and experiences"
    print(f"Testing query: {query}")
    try:
        result = await ask_health_assistant(query)
        # Detailed results from server response format
        sources = result.get('sources', [])
        print(f"Total sources: {len(sources)}")
        
        medical_count = 0
        community_count = 0
        
        for s in sources:
            stype = s.get('type', 'UNKNOWN').lower()
            url = s.get('url', 'NO_URL')
            score = s.get('score', 'N/A')
            
            if stype == "anecdotal":
                community_count += 1
            else:
                medical_count += 1
                
            print(f"- [{stype.upper()}] {url} ({score})")
            
        print(f"\nBreakdown: {medical_count} Medical, {community_count} Community")
        
        if medical_count > 0 and community_count > 0:
            print("SUCCESS: Mixed and matched sources found!")
        else:
            print("FAILED: Did not find a mix.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(verify())
