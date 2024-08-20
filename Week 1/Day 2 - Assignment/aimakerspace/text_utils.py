import os
from typing import List, Dict
import re


class TextFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".txt"):
            self.load_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .txt file."
            )

    def load_file(self):
        with open(self.path, "r", encoding=self.encoding) as f:
            self.documents.append(f.read())

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".txt"):
                    with open(
                        os.path.join(root, file), "r", encoding=self.encoding
                    ) as f:
                        self.documents.append(f.read())

    def load_documents(self):
        self.load()
        return self.documents


class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks


class DynamicHierarchicalTextSplitter:
    def __init__(
        self,
        file_path: str,
        encoding: str = "utf-8",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.file_path = file_path
        self.encoding = encoding
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents = []
        self.toc_sections = []
        self.toc_parts_by_section = {}

    def load_file(self):
        with open(self.file_path, "r", encoding=self.encoding) as f:
            self.documents.append(f.read())


    def infer_toc(self, lines: List[str]= None) -> List[str]:
        # Using while loop to iterate through the lines finding the first section of the TOC whchis is all in uppercase.
        # Once the first section is found, go through subsequent lines until another line with all uper case is found which is next section.
        # Each lines below the sections starts with "Part <part number>:"
        # Each part lines will have "Part <part number>: <part title> <page number>"
        # Now add the section and its part details as object in self.toc_parts_by_section
        # Return the list of sections found in the TOC
        lines = self.documents[0].splitlines()
        current_section = None
        first_section = None
        sections = []
        toc_pattern = re.compile(r'^Part \d+: .+? \d+$')
        non_part_pattern = re.compile(r'^.+? \d+$')
        incomplete_part = ""

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line == "THE PMARCA GUIDE TO HIRING":
                print("debug")

            # Check if the line is a section (all uppercase letters)
            if line.isupper() and not line.startswith("Part"):
                if line.isupper() and lines[i+1].strip().isupper(): # Means section title in two lines
                    line = line + " " + lines[i+1].strip()
                    i += 1
                if current_section:
                    # Store the previous section before starting a new one
                    self.toc_parts_by_section[line] = []
                current_section = line
                first_section = current_section if not first_section else first_section
                sections.append(current_section)
            elif current_section:
                # Append the current line to the incomplete part
                incomplete_part += " " + line if incomplete_part else line

                # Check if the current incomplete part matches either the complete "Part" pattern or the non-part pattern
                if toc_pattern.match(incomplete_part) or non_part_pattern.match(incomplete_part):
                    # Extract the part details
                    part_details = incomplete_part.strip()
                    if current_section in self.toc_parts_by_section:
                        self.toc_parts_by_section[current_section].append(part_details)
                    else:
                        self.toc_parts_by_section[current_section] = [part_details]
                    # Reset the incomplete part for the next line
                    incomplete_part = ""

                # Detect start of content after the last TOC section
                elif first_section.strip().lower() == incomplete_part.strip().lower():
                    # We detect that the TOC is complete
                    print("Detected start of content after TOC.")
                    break

            i += 1

        # Ensure the last section is stored if the loop ends with an incomplete part
        if incomplete_part and current_section:
            if current_section in self.toc_parts_by_section:
                self.toc_parts_by_section[current_section].append(incomplete_part.strip())
            else:
                self.toc_parts_by_section[current_section] = [incomplete_part.strip()]

        return sections
       

    def split_document_to_chunks(self):
        # Read the entire file content
        self.load_file()
        toc = self.infer_toc()

        # Define the regex pattern to identify the start of each part
        # part_pattern = re.compile(r"(Part \d+: .+?)(\d{1,4})")
        # parts = part_pattern.split(content)

        # # Initialize variables to track the current section and part
        # current_section = None
        # current_part_number = None
        # current_part_title = None
        # current_page_number = None

        # # Iterate through the parts and split into chunks
        # chunk = ""
        # chunk_index = 0

        # for i in range(0, len(parts), 3):
        #     if i == 0:
        #         # Set current section based on the initial content
        #         section_match = re.search(r"(THE PMARCA GUIDE TO .+?)\n", parts[i])
        #         if section_match:
        #             current_section = section_match.group(1)

        #     current_part_number = parts[i + 1].split(":")[0].strip()
        #     current_part_title = parts[i + 1].split(":")[1].strip()
        #     current_page_number = parts[i + 2].strip()

        #     header = f"{current_section}\n{current_part_number}: {current_part_title} - Page {current_page_number}\n\n"

        #     content_part = parts[i] + header + parts[i + 2]

        #     # If adding this part exceeds the max_chunk_size, save the current chunk and start a new one
        #     if len(chunk) + len(content_part) > max_chunk_size:
        #         with open(
        #             f"{output_dir}/chunk_{chunk_index}.txt", "w", encoding="utf-8"
        #         ) as chunk_file:
        #             chunk_file.write(chunk)
        #         chunk_index += 1
        #         chunk = header + content_part
        #     else:
        #         chunk += content_part

        # # Save any remaining content as the last chunk
        # if chunk:
        #     with open(
        #         f"{output_dir}/chunk_{chunk_index}.txt", "w", encoding="utf-8"
        #     ) as chunk_file:
        #         chunk_file.write(chunk)

        # print(f"Document split into {chunk_index + 1} chunks.")

    def infer_structure_and_split(self, text: str) -> Dict:
        lines = text.splitlines()

        current_chapter = None
        current_part = None
        current_page = None

        toc_found = False
        chapter_titles = []
        chapter_content = []

        for i, line in enumerate(
            lines[1:], start=1
        ):  # Start processing after the title
            line = line.strip()

            # Skip empty lines
            if line == "":
                continue

            # Identify the TOC section and extract chapter titles
            if "table of contents" in line.lower():
                toc_found = True
                continue
            if toc_found and not line:
                toc_found = False
            if toc_found and line:
                chapter_titles.append(line.strip())

            # Identify the post details (description, author, attributes)
            if i == 2:
                hierarchy["Post Details"]["Description"] = line
            elif i == 3:
                hierarchy["Post Details"]["Author"] = line
            elif 4 <= i <= 6:
                hierarchy["Post Details"]["Attributes"].append(line)

            # Infer chapters based on TOC titles and subsequent matching lines in the text
            if line in chapter_titles:
                if current_chapter:
                    # Store the previous chapter's content
                    current_chapter["Content"] = "\n".join(chapter_content)
                    chapter_content = []
                current_chapter = {"Chapter Title": line, "Parts": []}
                hierarchy["Post Details"]["Contents"]["Chapters"].append(
                    current_chapter
                )

            # Handle parts, pages, and content
            if "part " in line.lower() and ":" in line:
                part_number = line.split(" ")[1].rstrip(":")
                current_part = {
                    "Part Number": part_number,
                    "Pages": [],
                    "Title": lines[i - 1].strip(),
                }
                if current_chapter is not None:
                    current_chapter["Parts"].append(current_part)
            elif "page " in line.lower() and ":" in line:
                current_page = {"Page": line, "Chunks": []}
                if current_part is not None:
                    current_part["Pages"].append(current_page)
            else:
                # Add content to the current chapter, page, or part
                if current_page is not None:
                    chunks = self.chunk_text(line)
                    current_page["Chunks"].extend(chunks)
                elif current_part is not None:
                    chunks = self.chunk_text(line)
                    current_part.setdefault(
                        "Pages", [{"Page": "Implicit", "Chunks": chunks}]
                    )
                elif current_chapter is not None:
                    chapter_content.append(line)
                else:
                    chunks = self.chunk_text(line)
                    hierarchy["Post Details"]["Contents"]["Chapters"].append(
                        {
                            "Chapter Title": "Implicit",
                            "Parts": [
                                {
                                    "Part Number": "Implicit",
                                    "Pages": [{"Page": "Implicit", "Chunks": chunks}],
                                }
                            ],
                        }
                    )

        # Handle the last chapter's content
        if current_chapter:
            current_chapter["Content"] = "\n".join(chapter_content)

        return hierarchy

    def chunk_text(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def normalize_hierarchy(self, hierarchy: Dict) -> List[Dict]:
        """Flatten the hierarchy into a list of chunks with attributes."""
        normalized_chunks = []
        for chapter in hierarchy["Post Details"]["Contents"]["Chapters"]:
            chapter_title = chapter["Chapter Title"]
            for part in chapter["Parts"]:
                part_number = part["Part Number"]
                part_title = part["Title"]
                for page in part["Pages"]:
                    page_title = page["Page"]
                    for chunk in page["Chunks"]:
                        normalized_chunks.append(
                            {
                                "Chapter": chapter_title,
                                "Part Number": part_number,
                                "Part Title": part_title,
                                "Page": page_title,
                                "Chunk": chunk,
                            }
                        )
        return normalized_chunks


def using_text_char_splitter():
    loader = TextFileLoader("data/KingLear.txt")
    loader.load()
    splitter = CharacterTextSplitter()
    chunks = splitter.split_texts(loader.documents)
    print(len(chunks))
    print(chunks[0])
    print("--------")
    print(chunks[1])
    print("--------")
    print(chunks[-2])
    print("--------")
    print(chunks[-1])


if __name__ == "__main__":
    loader = TextFileLoader("data/PMarcaBlogs.txt")
    loader.load()
    splitter = HierarchicalTextSplitter()

    # Process documents hierarchically
    hierarchical_documents = []
    for document in loader.documents:
        hierarchy = splitter.split_text(document)
        normalized_chunks = splitter.normalize_hierarchy(hierarchy)
        hierarchical_documents.extend(normalized_chunks)

    print(len(hierarchical_documents))
    print(hierarchical_documents[0])
    print("--------")
    print(hierarchical_documents[1])
    print("--------")
    print(hierarchical_documents[-2])
    print("--------")
    print(hierarchical_documents[-1])
