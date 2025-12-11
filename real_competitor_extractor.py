# app/real_competitor_extractor.py
import requests
import os
import re
import time
from typing import List, Dict

class RealCompetitorExtractor:
    def __init__(self):
        # Get API key from environment variable
        self.serpapi_key = os.getenv('SERPAPI_KEY') 
        
    def search_with_serpapi(self, startup_idea: str) -> List[Dict]:
        """Use SerpAPI to get real Google search results"""
        try:
            # Check if API key is available
            if not self.serpapi_key or self.serpapi_key.startswith('your_'):
                print("❌ No valid SerpAPI key found")
                return []
                
            print(f"🔍 Searching SerpAPI for: {startup_idea}")
                
            params = {
                'q': f'{startup_idea} competitors similar companies',
                'api_key': self.serpapi_key,
                'num': 6
            }
            
            response = requests.get('https://serpapi.com/search', params=params, timeout=15)
            data = response.json()
            
            competitors = []
            if 'organic_results' in data:
                for result in data['organic_results'][:5]:
                    title = result.get('title', '')
                    snippet = result.get('snippet', '')
                    
                    # Skip comparison articles and irrelevant results
                    if any(word in title.lower() for word in ['vs', 'comparison', 'alternative', 'review']):
                        continue
                    
                    company_name = self.clean_company_name(title)
                    if company_name and len(company_name) > 2:
                        competitors.append({
                            'name': company_name,
                            'url': result.get('link', ''),
                            'description': snippet[:100] if snippet else f"Company in {startup_idea} space",
                            'source': 'Google Search',
                            'relevance': 'Market competitor'
                        })
            
            print(f"✅ Found {len(competitors)} competitors via SerpAPI")
            return competitors
            
        except Exception as e:
            print(f"❌ SerpAPI search failed: {e}")
            return []

    def clean_company_name(self, title: str) -> str:
        """Extract clean company name from search result title"""
        if not title:
            return ""
            
        # Remove common suffixes and prefixes
        removals = [
            ' - Wikipedia', '| Crunchbase', '| LinkedIn', 
            '·', '–', '—', '...', ' | ', ' - '
        ]
        
        clean_title = title
        for removal in removals:
            clean_title = clean_title.split(removal)[0]
        
        # Remove content in parentheses
        clean_title = re.sub(r'\([^)]*\)', '', clean_title)
        
        # Take only the main part (before common separators)
        separators = [' - ', ' | ', ' · ', ' – ', ' — ']
        for sep in separators:
            if sep in clean_title:
                clean_title = clean_title.split(sep)[0]
        
        return clean_title.strip()

    def duckduckgo_fallback(self, startup_idea: str) -> List[Dict]:
        """Fallback using DuckDuckGo (no API key needed)"""
        try:
            print("🦆 Trying DuckDuckGo fallback...")
            url = "https://api.duckduckgo.com/"
            params = {
                'q': f'{startup_idea} companies startups',
                'format': 'json',
                'no_html': 1,
                'skip_disambig': 1
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            competitors = []
            
            # Extract from RelatedTopics
            if 'RelatedTopics' in data:
                for topic in data['RelatedTopics'][:4]:
                    text = topic.get('Text', '')
                    if text:
                        # Look for company names (capitalized multi-word phrases)
                        potential_companies = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
                        for company in potential_companies:
                            if (len(company) > 3 and 
                                company not in ['The', 'This', 'For', 'With', 'How'] and
                                not any(word in company.lower() for word in ['wikipedia', 'homepage'])):
                                competitors.append({
                                    'name': company,
                                    'description': text[:80] + '...' if len(text) > 80 else text,
                                    'source': 'DuckDuckGo',
                                    'relevance': 'Related company'
                                })
            
            # Extract from Abstract
            if 'Abstract' in data and data['Abstract']:
                abstract = data['Abstract']
                companies = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', abstract)
                for company in companies[:3]:
                    if len(company) > 3 and company not in [comp['name'] for comp in competitors]:
                        competitors.append({
                            'name': company,
                            'description': abstract[:80] + '...',
                            'source': 'DuckDuckGo',
                            'relevance': 'Mentioned in search results'
                        })
            
            print(f"✅ Found {len(competitors)} competitors via DuckDuckGo")
            return competitors
            
        except Exception as e:
            print(f"❌ DuckDuckGo fallback failed: {e}")
            return []

    def get_basic_fallback(self, startup_idea: str) -> List[Dict]:
        """Basic fallback when no API results"""
        print("📋 Using basic fallback competitors")
        
        # Simple keyword-based generic competitors
        idea_lower = startup_idea.lower()
        
        if any(word in idea_lower for word in ['ai', 'artificial', 'machine learning']):
            return [{
                'name': 'AI Technology Companies',
                'description': 'Various companies working in artificial intelligence',
                'source': 'Market Research',
                'relevance': 'AI industry competition'
            }]
        elif any(word in idea_lower for word in ['app', 'mobile', 'software']):
            return [{
                'name': 'Tech Startups',
                'description': 'Technology companies in software and applications',
                'source': 'Market Research', 
                'relevance': 'Software market competition'
            }]
        else:
            return [{
                'name': 'Industry Players',
                'description': 'Established companies in this market space',
                'source': 'Market Research',
                'relevance': 'General market competition'
            }]

    def find_real_competitors(self, startup_idea: str) -> List[Dict]:
        """MAIN FUNCTION: Extract real competitors from web sources"""
        print(f"🎯 Searching for competitors: {startup_idea}")
        
        all_competitors = []
        
        # Method 1: SerpAPI (Primary - requires API key)
        serp_competitors = self.search_with_serpapi(startup_idea)
        all_competitors.extend(serp_competitors)
        
        # Method 2: DuckDuckGo fallback (if no SerpAPI results)
        if not all_competitors:
            ddg_competitors = self.duckduckgo_fallback(startup_idea)
            all_competitors.extend(ddg_competitors)
        
        # Method 3: Basic fallback (if still no results)
        if not all_competitors:
            basic_competitors = self.get_basic_fallback(startup_idea)
            all_competitors.extend(basic_competitors)
        
        # Remove duplicates
        unique_competitors = []
        seen_names = set()
        
        for comp in all_competitors:
            name = comp.get('name', '').lower().strip()
            if name and name not in seen_names and len(name) > 2:
                seen_names.add(name)
                unique_competitors.append(comp)
        
        print(f"🎉 Total unique competitors found: {len(unique_competitors)}")
        return unique_competitors[:5]  # Return top 5

# Global instance
competitor_extractor = RealCompetitorExtractor()