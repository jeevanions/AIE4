
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Union
from urllib.parse import urlparse
import logging
from abc import ABC
from pathlib import Path
import tempfile
import os
import requests
import re

from langchain.document_loaders import PyMuPDFLoader
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.parsers import PyMuPDFParser
from langchain.schema import Document as LCDocument  # Updated import
from langchain_core.documents import Document
from langchain_core.utils import get_from_dict_or_env
from langchain_community.document_loaders.blob_loaders import Blob
from pydantic import BaseModel

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import ConversionStatus, PipelineOptions

import pymupdf4llm

# logger = logging.getLogger(__file__)

#================================================================================================
def convert_pdfs_to_markdown(path: str):
    # List to hold the markdown contents of all PDFs
    markdown_contents = []
    md_langchain = []

    if os.path.isfile(path) and path.endswith(".pdf"):
        # Single PDF file case
        file_path = path
        try:
            # Convert the PDF to markdown
            md_text = pymupdf4llm.to_markdown(file_path, page_chunks=True)
            markdown_contents.append(md_text)
        except Exception as e:
            print(f"Failed to convert {file_path}: {e}")
    else:
        # Directory case
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".pdf"):
                    file_path = os.path.join(root, file)
                    print(file_path)
                    try:
                        # Convert the PDF to markdown
                        md_text = pymupdf4llm.to_markdown(file_path, page_chunks=True)
                        markdown_contents.append(md_text)
                    except Exception as e:
                        print(f"Failed to convert {file_path}: {e}")

    # Convert the markdown contents to LangChain format
    for file in markdown_contents:
        langchain_output = llamaindex_to_langchain(file)
        md_langchain.extend(langchain_output)

    return md_langchain


class Document:
    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"Document(page_content={repr(self.page_content)}, metadata={self.metadata})"


def llamaindex_to_langchain(llamaindex_output: List[Dict[str, Any]]) -> List[Document]:
    langchain_output = []

    for item in llamaindex_output:
        # Extract the text content and metadata from the LlamaIndex dictionary
        text = item["text"]
        metadata = item["metadata"]

        # Transform the metadata to match the LangChain structure
        langchain_metadata = {
            "source": metadata["file_path"],
            "file_path": metadata["file_path"],
            "page": metadata["page"],
            "total_pages": metadata["page_count"],
            "format": metadata["format"],
            "title": metadata["title"],
            "author": metadata["author"],
            "subject": metadata["subject"],
            "keywords": metadata["keywords"],
            "creator": metadata["creator"],
            "producer": metadata["producer"],
            "creationDate": metadata["creationDate"],
            "modDate": metadata["modDate"],
            "trapped": metadata["trapped"],
        }

        # Create a new Document object for the LangChain output
        document = Document(page_content=text, metadata=langchain_metadata)
        langchain_output.append(document)

    return langchain_output

#================================================================================================
class BasePDFLoader(BaseLoader, ABC):
    """Base Loader class for `PDF` files.

    If the file is a web path, it will download it to a temporary file, use it, then
        clean up the temporary file after completion.
    """

    def __init__(self, file_path: Union[str, Path], *, headers: Optional[Dict] = None):
        """Initialize with a file path.

        Args:
            file_path: Either a local, S3 or web path to a PDF file.
            headers: Headers to use for GET request to download a file from a web path.
        """
        self.file_path = str(file_path)
        self.web_path = None
        self.headers = headers
        if "~" in self.file_path:
            self.file_path = os.path.expanduser(self.file_path)

        # If the file is a web path or S3, download it to a temporary file, and use that
        if not os.path.isfile(self.file_path) and self._is_valid_url(self.file_path):
            self.temp_dir = tempfile.TemporaryDirectory()
            _, suffix = os.path.splitext(self.file_path)
            if self._is_s3_presigned_url(self.file_path):
                suffix = urlparse(self.file_path).path.split("/")[-1]
            temp_pdf = os.path.join(self.temp_dir.name, f"tmp{suffix}")
            self.web_path = self.file_path
            if not self._is_s3_url(self.file_path):
                r = requests.get(self.file_path, headers=self.headers)
                if r.status_code != 200:
                    raise ValueError(
                        "Check the url of your file; returned status code %s"
                        % r.status_code
                    )

                with open(temp_pdf, mode="wb") as f:
                    f.write(r.content)
                self.file_path = str(temp_pdf)
        elif not os.path.isfile(self.file_path):
            raise ValueError("File path %s is not a valid file or url" % self.file_path)

    def __del__(self) -> None:
        if hasattr(self, "temp_dir"):
            self.temp_dir.cleanup()

    @staticmethod
    def _is_valid_url(url: str) -> bool:
        """Check if the url is valid."""
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)

    @staticmethod
    def _is_s3_url(url: str) -> bool:
        """check if the url is S3"""
        try:
            result = urlparse(url)
            if result.scheme == "s3" and result.netloc:
                return True
            return False
        except ValueError:
            return False

    @staticmethod
    def _is_s3_presigned_url(url: str) -> bool:
        """Check if the url is a presigned S3 url."""
        try:
            result = urlparse(url)
            return bool(re.search(r"\.s3\.amazonaws\.com$", result.netloc))
        except ValueError:
            return False

    @property
    def source(self) -> str:
        return self.web_path if self.web_path is not None else self.file_path


class PyMuPDF4LLMLoader(BasePDFLoader, ABC):
    """Load `PDF` files using `pymupdf4llm` for enhanced processing.

    This loader leverages `pymupdf4llm` to extract text and metadata from PDF files,
    potentially integrating language model capabilities for improved analysis.

    Args:
        file_path (str): Path to the PDF file to load.
        headers (Optional[Dict]): Headers to use if the file is accessed via a web URL.
        extract_images (bool): Whether to extract images from the PDF.
        llm_model (Optional[str]): Identifier for the language model to use.
        **kwargs: Additional keyword arguments for `pymupdf4llm` parser.

    Example:
        loader = PyMuPDF4LLMLoader(
            file_path="example.pdf",
            extract_images=True,
        )
        documents = loader.load()
        for doc in documents:
            print(doc.page_content)
            print(doc.metadata)
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        *,
        headers: Optional[Dict] = None,
        extract_images: bool = False,
        llm_model: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the loader with the given file path and options."""
        super().__init__(file_path, headers=headers)
        self.extract_images = extract_images
        self.parser = PyMuPDFParser(
            extract_images=self.extract_images,
            **kwargs,
        )

    def load(self) -> List[Document]:
        """Load the PDF and return a list of Document objects."""
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load the PDF, yielding Document objects one by one."""
        if self.web_path:
            local_path = self._download_pdf(self.web_path)
        else:
            local_path = self.file_path
        print(f"Loading PDF from {local_path}...")
        loader = pymupdf4llm.to_markdown(local_path, page_chunks=True, write_images=False)
        documents = llamaindex_to_langchain(loader)
        yield from documents

    def __del__(self) -> None:
        """Cleanup resources if necessary."""
        if hasattr(self, "temp_dir"):
            self.temp_dir.cleanup()
    
    def _download_pdf(self, url: str) -> str:
        """Download PDF from a URL to a temporary file and return the file path."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            with open(temp_file.name, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return temp_file.name
        except Exception as e:
            raise ValueError(f"Error downloading PDF from {url}: {e}")

#================================================================================================


class DocumentMetadata(BaseModel):
    dl_doc_hash: str
    # source: str  # Uncomment if needed


class PDFLoader(BaseLoader):
    class LoaderType(str, Enum):
        PYMUPDF = "pymupdf"
        PYMUPDF4LLM = "pymupdf4llm"
        DOCLING = "docling"

    class ParseType(str, Enum):
        MARKDOWN = "markdown"
        # JSON = "json"  # Uncomment if needed

    def __init__(
        self,
        file_path: List[str],
        loader_type: LoaderType = LoaderType.PYMUPDF,
        parse_type: Optional[ParseType] = None,
    ):
        self._file_paths = file_path if isinstance(file_path, list) else [file_path]
        self._loader_type = loader_type

        if parse_type is None and self._loader_type in {
            self.LoaderType.DOCLING,
            self.LoaderType.PYMUPDF4LLM,
        }:
            self._parse_type = self.ParseType.MARKDOWN
        else:
            self._parse_type = parse_type

        self._converter = None
        if self._loader_type == self.LoaderType.DOCLING:
            pipeline_options = PipelineOptions()
            pipeline_options.do_ocr=True
            pipeline_options.do_table_structure=True
            pipeline_options.table_structure_options.do_cell_matching = True

            doc_converter = DocumentConverter(
                pipeline_options=pipeline_options,
                pdf_backend=PyPdfiumDocumentBackend,
            )
            self._converter = doc_converter

    def lazy_load(self) -> Iterator[LCDocument]:
        print(f"Loading {len(self._file_paths)} documents with {self._loader_type} loader...")
        if self._loader_type == self.LoaderType.PYMUPDF4LLM:
            for source in self._file_paths:
                temp_file_path = None
                try:
                    loader = PyMuPDF4LLMLoader(source)
                    docs = loader.load()
                    for doc in docs:
                        yield doc
                except Exception as e:
                    print(f"Error loading document from {source}: {e}")
                    continue

                finally:
                    # Clean up temporary file if it was created
                    if temp_file_path and os.path.exists(temp_file_path):
                        try:
                            os.remove(temp_file_path)
                        except Exception as cleanup_error:
                            print(f"Error deleting temporary file {temp_file_path}: {cleanup_error}")
        elif self._loader_type == self.LoaderType.PYMUPDF:
            for source in self._file_paths:
                try:
                    loader = PyMuPDFLoader(source)
                    docs = loader.load()
                    for doc in docs:
                        yield doc
                except Exception as e:
                    print(f"Error loading with PyMuPDFLoader for {source}: {e}")
        elif self._loader_type == self.LoaderType.DOCLING:
            if not self._converter:
                raise RuntimeError("DocumentConverter is not initialized for Docling loader.")
            for source in self._file_paths:
                try:
                    dl_doc = self._converter.convert_single(source).output
                    if self._parse_type == self.ParseType.MARKDOWN:
                        text = dl_doc.export_to_markdown()
                    # elif self._parse_type == self.ParseType.JSON:
                    #     text = dl_doc.model_dump_json()
                    else:
                        raise RuntimeError(
                            f"Unexpected parse type encountered: {self._parse_type}"
                        )

                    lc_doc = LCDocument(
                        page_content=text,
                        metadata=DocumentMetadata(
                            dl_doc_hash=dl_doc.file_info.document_hash,
                        ).model_dump(),  # Changed to .dict() for compatibility
                    )
                    yield lc_doc

                except Exception as e:
                    print(f"Error loading with Docling for {source}: {e}")

        else:
            raise ValueError(f"11Unsupportedjeeva loader type: {self._loader_type}")

