"""
Research Paper Crawler and Analysis System for AlgoForge 3.0
Automatically fetches and analyzes quantitative finance research papers from multiple sources
"""
import asyncio
import aiohttp
import json
import re
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from xml.etree import ElementTree as ET
from loguru import logger
from config import config

@dataclass
class ResearchPaper:
    """Research paper metadata and content"""
    id: str
    title: str
    authors: List[str]
    abstract: str
    published_date: datetime
    source: str  # arxiv, ssrn, etc.
    url: str
    keywords: List[str]
    full_text: Optional[str] = None
    relevance_score: float = 0.0
    trading_signals: List[str] = None

class ArxivCrawler:
    """ArXiv research paper crawler for quantitative finance papers"""
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def search_papers(self, query: str, max_results: int = 50, days_back: int = 30) -> List[ResearchPaper]:
        """Search for papers on ArXiv"""
        if not self.session:
            raise RuntimeError("Crawler not initialized. Use async context manager.")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Build search query
        search_query = f"cat:q-fin.* AND ({query})"
        
        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'lastUpdatedDate',
            'sortOrder': 'descending'
        }
        
        logger.info(f"Searching ArXiv for: {query}")
        
        try:
            async with self.session.get(self.base_url, params=params) as response:
                response.raise_for_status()
                xml_content = await response.text()
                
                papers = self._parse_arxiv_response(xml_content)
                
                # Filter by date
                recent_papers = [
                    paper for paper in papers 
                    if paper.published_date >= start_date
                ]
                
                logger.success(f"Found {len(recent_papers)} recent papers from ArXiv")
                return recent_papers
                
        except Exception as e:
            logger.error(f"Error searching ArXiv: {e}")
            return []
    
    def _parse_arxiv_response(self, xml_content: str) -> List[ResearchPaper]:
        """Parse ArXiv API XML response"""
        papers = []
        
        try:
            root = ET.fromstring(xml_content)
            
            # Define namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('atom:entry', ns):
                try:
                    # Extract basic information
                    paper_id = entry.find('atom:id', ns).text.split('/')[-1]
                    title = entry.find('atom:title', ns).text.strip()
                    abstract = entry.find('atom:summary', ns).text.strip()
                    
                    # Extract authors
                    authors = []
                    for author in entry.findall('atom:author', ns):
                        name = author.find('atom:name', ns)
                        if name is not None:
                            authors.append(name.text)
                    
                    # Extract publication date
                    published = entry.find('atom:published', ns).text
                    pub_date = datetime.fromisoformat(published.replace('Z', '+00:00'))
                    
                    # Extract URL
                    url = entry.find('atom:id', ns).text
                    
                    # Extract keywords from abstract and title
                    keywords = self._extract_keywords(title + " " + abstract)
                    
                    paper = ResearchPaper(
                        id=paper_id,
                        title=title,
                        authors=authors,
                        abstract=abstract,
                        published_date=pub_date,
                        source="arxiv",
                        url=url,
                        keywords=keywords,
                        trading_signals=[]
                    )
                    
                    papers.append(paper)
                    
                except Exception as e:
                    logger.warning(f"Error parsing paper entry: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error parsing ArXiv XML: {e}")
        
        return papers
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        trading_keywords = [
            'momentum', 'mean reversion', 'arbitrage', 'alpha', 'beta',
            'volatility', 'sharpe ratio', 'portfolio', 'risk management',
            'algorithmic trading', 'quantitative', 'backtesting', 'returns',
            'factor investing', 'market microstructure', 'high frequency',
            'options', 'derivatives', 'machine learning', 'neural network',
            'ensemble', 'optimization', 'regression', 'classification'
        ]
        
        text_lower = text.lower()
        found_keywords = []
        
        for keyword in trading_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords

class ResearchAnalyzer:
    """Analyzes research papers for trading signals and actionable insights"""
    
    def __init__(self, claude_analyst):
        self.claude_analyst = claude_analyst
    
    async def analyze_paper_for_trading_signals(self, paper: ResearchPaper) -> Dict[str, Any]:
        """Analyze a research paper for actionable trading signals"""
        
        prompt = f"""
        Analyze this quantitative finance research paper for actionable trading signals and strategies:
        
        Title: {paper.title}
        Authors: {', '.join(paper.authors)}
        Abstract: {paper.abstract}
        Keywords: {', '.join(paper.keywords)}
        
        Extract the following information in JSON format:
        {{
            "relevance_score": 0.85,  // 0-1 scale for trading relevance
            "key_findings": ["Finding 1", "Finding 2"],
            "trading_signals": ["Signal 1", "Signal 2"],
            "implementation_difficulty": "easy/medium/hard",
            "asset_classes": ["equities", "bonds", "commodities"],
            "timeframes": ["intraday", "daily", "weekly"],
            "risk_factors": ["Risk 1", "Risk 2"],
            "backtesting_requirements": ["Data requirement 1", "Data requirement 2"],
            "expected_performance": {{
                "potential_alpha": 0.15,  // Expected excess return
                "implementation_complexity": 0.6,  // 0-1 scale
                "data_requirements": 0.4  // 0-1 scale
            }}
        }}
        
        Focus on practical, implementable strategies that can be coded in QuantConnect.
        """
        
        try:
            analysis = {}
            
            async for message in self.claude_analyst.query(prompt=prompt, options=self.claude_analyst.options):
                if message.type == "text":
                    content = message.content
                    
                    # Extract JSON analysis
                    if "```json" in content:
                        json_start = content.find("```json") + 7
                        json_end = content.find("```", json_start)
                        json_content = content[json_start:json_end].strip()
                        try:
                            analysis = json.loads(json_content)
                        except json.JSONDecodeError:
                            logger.warning(f"Could not parse analysis JSON for paper {paper.id}")
            
            # Update paper with analysis results
            paper.relevance_score = analysis.get('relevance_score', 0.0)
            paper.trading_signals = analysis.get('trading_signals', [])
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing paper {paper.id}: {e}")
            return {}
    
    async def synthesize_research_insights(self, papers: List[ResearchPaper]) -> Dict[str, Any]:
        """Synthesize insights from multiple research papers"""
        
        # Filter for highly relevant papers
        relevant_papers = [p for p in papers if p.relevance_score >= 0.6]
        
        if not relevant_papers:
            return {}
        
        # Prepare synthesis prompt
        papers_summary = []
        for paper in relevant_papers[:10]:  # Limit to top 10 papers
            papers_summary.append({
                'title': paper.title,
                'key_signals': paper.trading_signals,
                'relevance': paper.relevance_score
            })
        
        prompt = f"""
        Synthesize actionable trading insights from these quantitative finance research papers:
        
        Papers Summary:
        {json.dumps(papers_summary, indent=2)}
        
        Provide synthesis in JSON format:
        {{
            "common_themes": ["Theme 1", "Theme 2"],
            "conflicting_findings": ["Conflict 1", "Conflict 2"],
            "implementation_priorities": [
                {{
                    "strategy": "Strategy name",
                    "supporting_papers": 3,
                    "implementation_difficulty": "medium",
                    "expected_impact": "high"
                }}
            ],
            "market_conditions": {{
                "bull_market_strategies": ["Strategy 1"],
                "bear_market_strategies": ["Strategy 2"],
                "neutral_market_strategies": ["Strategy 3"]
            }},
            "recommended_next_steps": ["Step 1", "Step 2"]
        }}
        """
        
        try:
            synthesis = {}
            
            async for message in self.claude_analyst.query(prompt=prompt, options=self.claude_analyst.options):
                if message.type == "text":
                    content = message.content
                    
                    if "```json" in content:
                        json_start = content.find("```json") + 7
                        json_end = content.find("```", json_start)
                        json_content = content[json_start:json_end].strip()
                        try:
                            synthesis = json.loads(json_content)
                        except json.JSONDecodeError:
                            logger.warning("Could not parse synthesis JSON")
            
            return synthesis
            
        except Exception as e:
            logger.error(f"Error synthesizing research insights: {e}")
            return {}

class ResearchPipeline:
    """Complete research pipeline for continuous paper analysis"""
    
    def __init__(self, claude_analyst):
        self.claude_analyst = claude_analyst
        self.analyzer = ResearchAnalyzer(claude_analyst)
        self.arxiv_crawler = ArxivCrawler()
    
    async def run_daily_research_update(self) -> Dict[str, Any]:
        """Run daily research update pipeline"""
        logger.info("🔬 Starting daily research update...")
        
        results = {
            'papers_found': 0,
            'papers_analyzed': 0,
            'actionable_insights': 0,
            'synthesis': {}
        }
        
        try:
            # Define search queries for different areas
            search_queries = [
                "momentum trading algorithmic",
                "mean reversion portfolio optimization",
                "machine learning trading signals",
                "risk management quantitative",
                "factor investing systematic",
                "volatility forecasting trading"
            ]
            
            all_papers = []
            
            # Crawl papers from each query
            async with self.arxiv_crawler as crawler:
                for query in search_queries:
                    papers = await crawler.search_papers(query, max_results=20, days_back=7)
                    all_papers.extend(papers)
            
            # Remove duplicates
            unique_papers = {}
            for paper in all_papers:
                if paper.id not in unique_papers:
                    unique_papers[paper.id] = paper
            
            all_papers = list(unique_papers.values())
            results['papers_found'] = len(all_papers)
            
            logger.info(f"Found {len(all_papers)} unique papers")
            
            # Analyze each paper for trading signals
            analyzed_papers = []
            for paper in all_papers:
                analysis = await self.analyzer.analyze_paper_for_trading_signals(paper)
                if analysis and paper.relevance_score >= 0.5:
                    analyzed_papers.append(paper)
            
            results['papers_analyzed'] = len(analyzed_papers)
            results['actionable_insights'] = len([p for p in analyzed_papers if p.relevance_score >= 0.7])
            
            # Synthesize insights
            if analyzed_papers:
                synthesis = await self.analyzer.synthesize_research_insights(analyzed_papers)
                results['synthesis'] = synthesis
            
            logger.success(f"✅ Research update complete: {results['actionable_insights']} actionable insights found")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in research pipeline: {e}")
            return results

# Example usage and testing
async def test_research_pipeline():
    """Test the research pipeline"""
    from claude_integration import ClaudeQuantAnalyst
    
    claude_analyst = ClaudeQuantAnalyst()
    pipeline = ResearchPipeline(claude_analyst)
    
    # Run research update
    results = await pipeline.run_daily_research_update()
    
    logger.info("Research Pipeline Results:")
    logger.info(f"Papers found: {results['papers_found']}")
    logger.info(f"Papers analyzed: {results['papers_analyzed']}")
    logger.info(f"Actionable insights: {results['actionable_insights']}")
    
    if results['synthesis']:
        logger.info("Synthesis insights:")
        for theme in results['synthesis'].get('common_themes', []):
            logger.info(f"  - {theme}")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_research_pipeline())