import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import re
from typing import List, Dict

def search_arxiv_papers(keywords: List[str], days_back: int = 3, sort_by: str = 'lastUpdatedDate', filter_by: str = 'updated') -> List[Dict]:
    """
    Search arXiv papers for specific keywords in the past N days
    
    Args:
        keywords: List of keywords to search for
        days_back: Number of days to look back (default: 3)
    
    Returns:
        List of dictionaries containing paper information
    """
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # Format dates for arXiv API (YYYYMMDD format)
    start_date_str = start_date.strftime("%Y%m%d")
    end_date_str = end_date.strftime("%Y%m%d")
    
    # Build search query - combine keywords with OR
    search_terms = " OR ".join([f'"{keyword}"' for keyword in keywords])
    
    # arXiv API parameters
    base_url = "http://export.arxiv.org/api/query"
    params = {
        'search_query': f'all:({search_terms})',
        'start': 0,
        'max_results': 100,  # Adjust as needed
        'sortBy': sort_by,
        'sortOrder': 'descending'
    }
    
    print(f"Searching arXiv for papers containing: {', '.join(keywords)}")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} (based on {filter_by})")
    print(f"Sorted by: {sort_by} (descending)")
    print("-" * 60)
    
    try:
        # Make API request
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(response.content)
        
        # Define namespace
        ns = {'atom': 'http://www.w3.org/2005/Atom',
              'arxiv': 'http://arxiv.org/schemas/atom'}
        
        papers = []
        entries = root.findall('atom:entry', ns)
        
        for entry in entries:
            # Extract paper information
            title = entry.find('atom:title', ns).text.strip()
            summary = entry.find('atom:summary', ns).text.strip()
            
            # Get submission and updated dates
            published = entry.find('atom:published', ns).text
            updated_elem = entry.find('atom:updated', ns)
            updated = updated_elem.text if updated_elem is not None else published
            pub_date = datetime.strptime(published[:10], '%Y-%m-%d')
            upd_date = datetime.strptime(updated[:10], '%Y-%m-%d')
            
            # Filter by selected date field
            date_to_check = upd_date if filter_by == 'updated' else pub_date
            if start_date.date() <= date_to_check.date() <= end_date.date():
                # Get authors
                authors = []
                for author in entry.findall('atom:author', ns):
                    name = author.find('atom:name', ns).text
                    authors.append(name)
                
                # Get arXiv ID and URL
                arxiv_id = entry.find('atom:id', ns).text.split('/')[-1]
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                abs_url = f"https://arxiv.org/abs/{arxiv_id}"
                
                # Check which keywords are found
                found_keywords = []
                text_to_search = (title + " " + summary).lower()
                
                for keyword in keywords:
                    if keyword.lower() in text_to_search:
                        found_keywords.append(keyword)
                
                if found_keywords:  # Only include if keywords are found
                    # Try to extract organizations from author affiliations first
                    organizations_from_authors = extract_organizations_from_authors(entry, ns)
                    
                    paper_info = {
                        'title': title,
                        'authors': authors,
                        'arxiv_id': arxiv_id,
                        'published_date': pub_date.strftime('%Y-%m-%d'),
                        'updated_date': upd_date.strftime('%Y-%m-%d'),
                        'abstract': summary,
                        'pdf_url': pdf_url,
                        'abstract_url': abs_url,
                        'found_keywords': found_keywords,
                        'organizations_from_authors': organizations_from_authors
                    }
                    papers.append(paper_info)
        
        return papers
        
    except requests.RequestException as e:
        print(f"Error fetching data from arXiv: {e}")
        return []
    except ET.ParseError as e:
        print(f"Error parsing XML response: {e}")
        return []

def extract_organizations_from_authors(entry, ns):
    """Extract organization information from author affiliations in arXiv entry"""
    organizations = set()
    
    # Look for arxiv:affiliation in author entries
    for author in entry.findall('atom:author', ns):
        affiliation = author.find('arxiv:affiliation', ns)
        if affiliation is not None and affiliation.text:
            org = affiliation.text.strip()
            if len(org) > 3:  # Filter out very short affiliations
                organizations.add(org)
    
    return ", ".join(list(organizations)[:3]) if organizations else "Not specified"

def extract_organizations(abstract: str, title: str = "") -> str:
    """Extract organization information from abstract and title with improved accuracy"""
    text_to_search = (abstract + " " + title).lower()
    
    # More specific patterns for organizations
    org_patterns = [
        # Universities - more specific patterns
        r'\b([A-Z][a-zA-Z\s]{2,30}?\s+University)\b',
        r'\b(University\s+of\s+[A-Z][a-zA-Z\s]{2,20})\b',
        r'\b([A-Z][a-zA-Z\s]{2,20}?\s+Institute\s+of\s+Technology)\b',
        
        # Specific well-known institutions
        r'\b(MIT|Stanford|Harvard|CMU|Berkeley|UCLA|USC|NYU|Caltech|Princeton|Yale|Columbia)\b',
        r'\b(ETH\s+Zurich|EPFL|Oxford|Cambridge|Imperial\s+College|UCL)\b',
        r'\b(Tsinghua|Peking\s+University|SJTU|NUS|NTU)\b',
        
        # Tech companies
        r'\b(Google|Microsoft|Meta|Apple|Amazon|NVIDIA|Intel|Adobe|IBM|OpenAI|DeepMind)\b',
        r'\b(Facebook|Tesla|Uber|ByteDance|Baidu|Tencent|Alibaba)\b',
        
        # Research institutions
        r'\b([A-Z][a-zA-Z\s]{2,25}?\s+Research\s+(?:Institute|Center|Lab))\b',
        r'\b([A-Z][a-zA-Z\s]{2,25}?\s+National\s+Laboratory)\b',
        
        # More specific institute patterns
        r'\b(Max\s+Planck\s+Institute)\b',
        r'\b(INRIA|CNRS|CEA)\b'
    ]
    
    organizations = set()
    for pattern in org_patterns:
        matches = re.findall(pattern, text_to_search, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0] if match[0] else match[1]
            
            # Clean up the match
            org = match.strip()
            if len(org) > 2 and not any(word in org.lower() for word in ['our', 'we', 'this', 'these', 'results', 'code', 'availab']):
                organizations.add(org)
    
    return ", ".join(list(organizations)[:3]) if organizations else "Not specified"

def display_results_markdown(papers: List[Dict]):
    """Display search results in markdown format"""
    
    if not papers:
        print("No papers found matching the criteria.")
        return
    
    # Get current date
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"**📅Last updated: {current_date}📅**")
    print(f"Total papers: {len(papers)}")
    print()  # Add blank line after header
    
    for paper in papers:
        # Try author affiliations first, then fall back to abstract extraction
        organizations = paper.get('organizations_from_authors', 'Not specified')
        if organizations == 'Not specified':
            organizations = extract_organizations(paper['abstract'], paper['title'])
        
        # Format authors
        authors_str = ", ".join(paper['authors'])
        updated_str = paper.get('updated_date', paper.get('published_date', ''))
        
        print(f"📄 **{paper['title']}**, Updated: {updated_str}, Organizations: {organizations}, Authors: {authors_str}, Link: {paper['abstract_url']}")
        print()  # Add blank line after each paper

def display_results(papers: List[Dict]):
    """Display search results in a formatted way"""
    
    if not papers:
        print("No papers found matching the criteria.")
        return
    
    print(f"\nFound {len(papers)} paper(s) matching your keywords:")
    print("=" * 80)
    
    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. {paper['title']}")
        print(f"   Authors: {', '.join(paper['authors'][:3])}" + 
              (f" (and {len(paper['authors']) - 3} more)" if len(paper['authors']) > 3 else ""))
        print(f"   Published: {paper['published_date']}")
        if 'updated_date' in paper:
            print(f"   Updated: {paper['updated_date']}")
        print(f"   arXiv ID: {paper['arxiv_id']}")
        print(f"   Keywords found: {', '.join(paper['found_keywords'])}")
        print(f"   Abstract URL: {paper['abstract_url']}")
        print(f"   PDF URL: {paper['pdf_url']}")
        
        # Show first 200 characters of abstract
        abstract_preview = paper['abstract'][:200] + "..." if len(paper['abstract']) > 200 else paper['abstract']
        print(f"   Abstract: {abstract_preview}")
        print("-" * 80)

def save_results_to_markdown(papers: List[Dict], filename: str = "arxiv_3dgs_papers.md"):
    """Save results to a markdown file"""
    
    if not papers:
        return
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"**📅Last updated: {current_date}📅**\n")
        f.write(f"Total papers: {len(papers)}\n\n")  # Add blank line after header
        
        for paper in papers:
            # Try author affiliations first, then fall back to abstract extraction
            organizations = paper.get('organizations_from_authors', 'Not specified')
            if organizations == 'Not specified':
                organizations = extract_organizations(paper['abstract'], paper['title'])
            
            # Format authors
            authors_str = ", ".join(paper['authors'])
            updated_str = paper.get('updated_date', paper.get('published_date', ''))
            
            f.write(f"📄 **{paper['title']}**, Updated: {updated_str}, {authors_str}, **Link**: {paper['abstract_url']}\n\n")  # Add blank line after each paper
    
    print(f"\nMarkdown results saved to {filename}")

def save_results_to_file(papers: List[Dict], filename: str = "arxiv_3dgs_papers.txt"):
    """Save results to a text file"""
    
    if not papers:
        return
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"ArXiv Search Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Keywords: 3DGS, Gaussian Splatting, Gaussian\n")
        f.write(f"Total papers found: {len(papers)}\n")
        f.write("=" * 80 + "\n\n")
        
        for i, paper in enumerate(papers, 1):
            f.write(f"{i}. {paper['title']}\n")
            f.write(f"   Authors: {', '.join(paper['authors'])}\n")
            f.write(f"   Published: {paper['published_date']}\n")
            if 'updated_date' in paper:
                f.write(f"   Updated: {paper['updated_date']}\n")
            f.write(f"   arXiv ID: {paper['arxiv_id']}\n")
            f.write(f"   Keywords found: {', '.join(paper['found_keywords'])}\n")
            f.write(f"   Abstract URL: {paper['abstract_url']}\n")
            f.write(f"   PDF URL: {paper['pdf_url']}\n")
            f.write(f"   Abstract: {paper['abstract']}\n")
            f.write("-" * 80 + "\n\n")
    
    print(f"\nDetailed results saved to {filename}")

# Main execution
if __name__ == "__main__":
    # Define keywords to search for
    keywords = ["3DGS", "Gaussian Splatting"]
    
    # Search for papers (sorted and filtered by last updated date by default)
    papers = search_arxiv_papers(keywords, days_back=3, sort_by='lastUpdatedDate', filter_by='updated')
    
    # Display results in markdown format
    print("\n" + "="*60)
    print("MARKDOWN FORMAT OUTPUT:")
    print("="*60)
    display_results_markdown(papers)
    
    # Display detailed results
    print("\n" + "="*60)
    print("DETAILED RESULTS:")
    print("="*60)
    display_results(papers)
    
    # Save results to files
    if papers:
        save_results_to_markdown(papers)  # Save markdown format
        save_results_to_file(papers)      # Save detailed format
    
    print(f"\nSearch completed. Found {len(papers)} relevant papers.")