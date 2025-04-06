from bs4 import BeautifulSoup

import time

from core.db.documents import NiceDocument
from crawlers.base import BaseAbstractCrawler

class NiceCrawler(BaseAbstractCrawler):
    model = NiceDocument

    def extract(self, link: str, **kwargs) -> None:
        # Check if this URL already exists in the database
        existing_doc = self.model.find(url=link)
        
        # Navigate to the detail page
        print(f"Navigating to: {link}")
        self.driver.get(link)
        time.sleep(1)  # Wait for page to load

        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        title_element = soup.find("h1", class_="page-header__heading")
        title = title_element.text.strip() if title_element else "Unknown Title"
        
        # Extract last updated date
        last_updated = None
        metadata = soup.find("ul", class_="page-header__metadata")
        if metadata:
            # Look for the "Last updated" list item
            for li in metadata.find_all("li"):
                if "Last updated" in li.text:
                    # Extract the datetime from the time element
                    time_element = li.find("time")
                    if time_element and time_element.has_attr("datetime"):
                        last_updated = time_element["datetime"]
                    else:
                        # If no datetime attribute, extract the text
                        last_updated = li.text.replace("Last updated:", "").strip()
                    break
        
        # Check if document already exists and if it needs updating
        if existing_doc:
            # Document exists - check if it needs updating
            if existing_doc.last_updated == last_updated:
                print(f"Document already exists and is up to date: {title}")
                return  # Skip processing
            else:
                print(f"Document exists but needs updating (last_updated changed from {existing_doc.last_updated} to {last_updated})")
                # Continue with extraction to update the document
        else:
            print(f"New document found: {title}")
        
        # Store basic info
        result_data = {"title": title, "url": link, "last_updated": last_updated, "chapters": []}
        
        # Extract chapters navigation
        detail_soup = BeautifulSoup(self.driver.page_source, "html.parser")
        nav_list = detail_soup.find("ul", class_="stacked-nav__list")
        
        if nav_list:
            chapter_links = nav_list.find_all("a")
            
            # Extract data from each chapter
            for chapter_link in chapter_links:
                chapter_url = "https://www.nice.org.uk" + chapter_link["href"]
                chapter_title = chapter_link.find("span", class_="stacked-nav__content-wrapper").text.strip()
                
                print(f"\tNavigating to chapter: {chapter_title}")
                self.driver.get(chapter_url)
                time.sleep(2)
                
                # Parse chapter content
                chapter_soup = BeautifulSoup(self.driver.page_source, "html.parser")
                
                # Initialize chapter data with simpler structure
                chapter_data = {
                    "title": chapter_title,
                    "url": chapter_url,
                    "markdown": ""  # Renamed to indicate markdown formatting
                }
                
                # Extract content from the main content div
                content_div = chapter_soup.find("div", attrs={"data-g": "12"})
                
                if content_div:
                    # Find the js-in-page-nav-target div
                    nav_target_div = content_div.find("div", class_="js-in-page-nav-target")
                    
                    if nav_target_div:
                        # For chapters other than overview, look for the chapter div
                        chapter_div = nav_target_div.find("div", class_="chapter")
                        
                        # If there's a chapter div, use it as content source, otherwise use the nav_target_div
                        content_source = chapter_div if chapter_div else nav_target_div
                        
                        # Add main title if present (could be in chapter div or nav_target_div)
                        main_title = content_source.find(["h1", "h2"], class_="title")
                        if main_title:
                            chapter_data["markdown"] += f"# {main_title.text.strip()}\n\n"
                        
                        # Extract all content in a hierarchical manner
                        sections = content_source.find_all(["div", "p", "h3", "h4"], recursive=False)
                        
                        # If no sections found at top level, check all elements
                        if not sections:
                            # Get all text with heading structure preserved
                            all_headings = content_source.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
                            for heading in all_headings:
                                level = int(heading.name[1])
                                markdown_heading = "#" * level
                                chapter_data["markdown"] += f"{markdown_heading} {heading.text.strip()}\n\n"
                                
                                # Get paragraphs and lists that follow until next heading
                                next_el = heading.find_next_sibling()
                                while next_el and not next_el.name.startswith("h"):
                                    if next_el.name == "p":
                                        # Extract links in paragraph and format them as markdown
                                        paragraph_text = next_el.text.strip()
                                        for link in next_el.find_all("a", href=True):
                                            link_text = link.text.strip()
                                            link_url = link["href"]
                                            if not link_url.startswith("http"):
                                                link_url = "https://www.nice.org.uk" + link_url
                                            # Replace the plain text with markdown link
                                            paragraph_text = paragraph_text.replace(link_text, f"[{link_text}]({link_url})")
                                        
                                        chapter_data["markdown"] += f"{paragraph_text}\n\n"
                                    elif next_el.name in ["ul", "ol"]:
                                        for i, li in enumerate(next_el.find_all("li")):
                                            if next_el.name == "ol":
                                                chapter_data["markdown"] += f"{i+1}. {li.text.strip()}\n"
                                            else:
                                                chapter_data["markdown"] += f"* {li.text.strip()}\n"
                                        chapter_data["markdown"] += "\n"
                                    next_el = next_el.find_next_sibling()
                        else:
                            # Process each section
                            for section in sections:
                                if section.name in ["h3", "h4"]:
                                    # It's a heading
                                    level = int(section.name[1])
                                    markdown_heading = "#" * level
                                    chapter_data["markdown"] += f"{markdown_heading} {section.text.strip()}\n\n"
                                elif section.name == "p":
                                    # It's a paragraph - convert links to markdown format
                                    paragraph_text = section.text.strip()
                                    for link in section.find_all("a", href=True):
                                        link_text = link.text.strip()
                                        link_url = link["href"]
                                        if not link_url.startswith("http"):
                                            link_url = "https://www.nice.org.uk" + link_url
                                        # Replace the plain text with markdown link
                                        paragraph_text = paragraph_text.replace(link_text, f"[{link_text}]({link_url})")
                                    
                                    chapter_data["markdown"] += f"{paragraph_text}\n\n"
                                elif section.name == "div" and section.get("class") and "section" in section.get("class"):
                                    # It's a nested section with a title
                                    section_title = section.find(["h3", "h4"], class_="title")
                                    if section_title:
                                        level = int(section_title.name[1])
                                        markdown_heading = "#" * level
                                        chapter_data["markdown"] += f"{markdown_heading} {section_title.text.strip()}\n\n"
                                    
                                    # Get all paragraphs and lists in this section
                                    for element in section.find_all(["p", "ul", "ol"]):
                                        if element.name == "p":
                                            # Format links in paragraph
                                            paragraph_text = element.text.strip()
                                            for link in element.find_all("a", href=True):
                                                link_text = link.text.strip()
                                                link_url = link["href"]
                                                if not link_url.startswith("http"):
                                                    link_url = "https://www.nice.org.uk" + link_url
                                                # Replace the plain text with markdown link
                                                paragraph_text = paragraph_text.replace(link_text, f"[{link_text}]({link_url})")
                                            
                                            chapter_data["markdown"] += f"{paragraph_text}\n\n"
                                        elif element.name in ["ul", "ol"]:
                                            for i, li in enumerate(element.find_all("li")):
                                                # Format list items with proper markdown
                                                li_text = li.text.strip()
                                                # Convert links in list items
                                                for link in li.find_all("a", href=True):
                                                    link_text = link.text.strip()
                                                    link_url = link["href"]
                                                    if not link_url.startswith("http"):
                                                        link_url = "https://www.nice.org.uk" + link_url
                                                    li_text = li_text.replace(link_text, f"[{link_text}]({link_url})")
                                                    
                                                if element.name == "ol":
                                                    chapter_data["markdown"] += f"{i+1}. {li_text}\n"
                                                else:
                                                    chapter_data["markdown"] += f"* {li_text}\n"
                                            chapter_data["markdown"] += "\n"
                    else:
                        # Fallback: just get all text from content_div with basic markdown formatting
                        all_text = content_div.get_text(separator="\n\n", strip=True)
                        chapter_data["markdown"] = all_text
                
                # Clean up any excessive newlines
                chapter_data["markdown"] = chapter_data["markdown"].replace("\n\n\n", "\n\n")
                
                result_data["chapters"].append(chapter_data)
        
        # Handle database update/create
        if existing_doc:
            # Update existing document
            existing_doc.title = result_data["title"]
            existing_doc.last_updated = result_data["last_updated"]
            existing_doc.chapters = result_data["chapters"]
            existing_doc.save(existing_doc=True)
            print(f"Updated existing document: {title}")
        else:
            # Create new document
            instance = self.model(
                title=result_data["title"],
                url=result_data["url"],
                last_updated=result_data["last_updated"],
                chapters=result_data["chapters"]
            )
            instance.save()
            print(f"Added new document: {title}")

        # # Save to file with proper date format
        # today = datetime.datetime.now().strftime("%Y-%m-%d")
        # with open(f"nice_crawler_data_{today}.json", "w", encoding="utf-8") as f:
        #     json.dump(result_data, f, ensure_ascii=False, indent=2)

    def extract_search_results(self, link: str, **kwargs) -> None:
        try:
            # Navigate to page
            self.driver.get(link)
            
            # Small delay to ensure page loads
            time.sleep(1)

            # Parse HTML
            soup = BeautifulSoup(self.driver.page_source, "html.parser")
            results = soup.find_all("li", class_="SearchCardList_listItem__4We2l")[:self.doc_limit]
            
            # Track processed and skipped documents
            processed_count = 0
            skipped_count = 0
            
            # Extract data
            for result in results:
                title_element = result.find("p", class_="card__heading").find("a")
                url = "https://www.nice.org.uk" + title_element["href"]
                result = self.extract(url)
                
                if result is None:
                    skipped_count += 1
                else:
                    processed_count += 1
            
            print(f"Completed extraction: {processed_count} updated/added, {skipped_count} skipped (already up to date)")
        except Exception as e:
            print(f"Error during extraction: {e}")