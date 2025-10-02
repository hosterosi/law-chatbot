import os
import io
import logging
import openai
import base64
from typing import Dict, List, Any
import pdf2image
from PIL import Image
import docx
import tempfile

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document processing and conversion to markdown"""

    def __init__(self):
        self.openai_client = None
        self._init_openai()

    def _init_openai(self):
        """Initialize OpenAI client"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not found")
            return

        try:
            self.openai_client = openai.OpenAI(api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")

    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encodings
            encodings = ["latin-1", "cp1252", "iso-8859-1"]
            for encoding in encodings:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            raise Exception("Unable to decode text file with any supported encoding")

    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = []
            for paragraph in doc.paragraphs:
                text.append(paragraph.text)
            return "\n".join(text)
        except Exception as e:
            raise Exception(f"Failed to extract text from DOCX: {e}")

    def extract_text_from_doc(self, file_path: str) -> str:
        """Extract text from DOC file (requires conversion)"""
        try:
            # Try using python-docx2txt as fallback
            import docx2txt

            return docx2txt.process(file_path)
        except ImportError:
            raise Exception("DOC file processing requires docx2txt package")
        except Exception as e:
            raise Exception(f"Failed to extract text from DOC: {e}")

    def is_scanned_pdf(self, file_path: str) -> bool:
        """Check if PDF is scanned (image-based) by trying to extract text"""
        try:
            import PyPDF2

            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages[:3]:  # Check first 3 pages
                    text += page.extract_text()

                # If extracted text is very short compared to expected content, likely scanned
                return len(text.strip()) < 100
        except Exception:
            # If we can't extract text, assume it's scanned
            return True

    def convert_pdf_to_images(self, file_path: str) -> List[Image.Image]:
        """Convert PDF pages to images"""
        try:
            images = pdf2image.convert_from_path(file_path)
            return images
        except Exception as e:
            raise Exception(f"Failed to convert PDF to images: {e}")

    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_data = buffer.getvalue()
        return base64.b64encode(image_data).decode("utf-8")

    def convert_image_to_markdown_with_gpt_vision(self, image: Image.Image) -> str:
        """Convert image to markdown using GPT-4 Vision (gpt-5-mini equivalent)"""
        if not self.openai_client:
            raise Exception("OpenAI client not initialized")

        try:
            # Convert image to base64
            image_b64 = self.image_to_base64(image)

            response = self.openai_client.chat.completions.create(
                model="gpt-5-mini",  # Using gpt-4o-mini as gpt-5-mini equivalent
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert document digitizer. Convert the image of a document page to clean, well-structured markdown format. Preserve the document structure, headings, lists, tables, and formatting. Extract all text accurately and maintain the original layout as much as possible in markdown format.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Convert this document page image to markdown format. Maintain structure and formatting.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"
                                },
                            },
                        ],
                    },
                ],
                max_tokens=4000,
                temperature=0,
            )

            return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"GPT Vision conversion failed: {e}")

    def convert_text_to_markdown_with_gpt(self, text: str) -> str:
        """Convert text to structured markdown using GPT-4"""
        if not self.openai_client:
            raise Exception("OpenAI client not initialized")

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5-nano",  # Using gpt-4o-mini as gpt-5-nano equivalent
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert document formatter. Convert the given text to clean, well-structured markdown format. Create appropriate headings, lists, tables, and formatting. Improve readability while preserving all information and meaning.",
                    },
                    {
                        "role": "user",
                        "content": f"Convert this text to well-structured markdown format:\n\n{text}",
                    },
                ],
                max_tokens=4000,
                temperature=0,
            )

            return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"GPT text conversion failed: {e}")

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            import PyPDF2

            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {e}")


def process_document(file_path: str, filename: str) -> Dict[str, Any]:
    """Main function to process document and convert to markdown"""
    processor = DocumentProcessor()

    try:
        # Get file extension
        file_extension = filename.rsplit(".", 1)[1].lower() if "." in filename else ""

        result = {
            "filename": filename,
            "file_type": file_extension,
            "success": False,
            "markdown_content": "",
            "processing_method": "",
            "pages": [],
        }

        if file_extension == "txt":
            # Process TXT file
            text = processor.extract_text_from_txt(file_path)
            markdown_content = processor.convert_text_to_markdown_with_gpt(text)

            result.update(
                {
                    "success": True,
                    "markdown_content": markdown_content,
                    "processing_method": "text_extraction_gpt_formatting",
                }
            )

        elif file_extension == "docx":
            # Process DOCX file
            text = processor.extract_text_from_docx(file_path)
            markdown_content = processor.convert_text_to_markdown_with_gpt(text)

            result.update(
                {
                    "success": True,
                    "markdown_content": markdown_content,
                    "processing_method": "docx_extraction_gpt_formatting",
                }
            )

        elif file_extension == "doc":
            # Process DOC file
            text = processor.extract_text_from_doc(file_path)
            markdown_content = processor.convert_text_to_markdown_with_gpt(text)

            result.update(
                {
                    "success": True,
                    "markdown_content": markdown_content,
                    "processing_method": "doc_extraction_gpt_formatting",
                }
            )

        elif file_extension == "pdf":
            # Check if PDF is scanned
            if processor.is_scanned_pdf(file_path):
                # Process as scanned PDF using vision
                logger.info("Processing scanned PDF with GPT Vision")
                images = processor.convert_pdf_to_images(file_path)

                page_markdowns = []
                for i, image in enumerate(images):
                    logger.info(f"Processing page {i+1}/{len(images)}")
                    page_markdown = processor.convert_image_to_markdown_with_gpt_vision(
                        image
                    )
                    page_markdowns.append(
                        {"page_number": i + 1, "markdown": page_markdown}
                    )

                # Combine all pages
                full_markdown = "\n\n---\n\n".join(
                    [p["markdown"] for p in page_markdowns]
                )

                result.update(
                    {
                        "success": True,
                        "markdown_content": full_markdown,
                        "processing_method": "scanned_pdf_gpt_vision",
                        "pages": page_markdowns,
                    }
                )

            else:
                # Process as text-based PDF
                logger.info("Processing text-based PDF")
                text = processor.extract_text_from_pdf(file_path)
                markdown_content = processor.convert_text_to_markdown_with_gpt(text)

                result.update(
                    {
                        "success": True,
                        "markdown_content": markdown_content,
                        "processing_method": "text_pdf_gpt_formatting",
                    }
                )

        else:
            raise Exception(f"Unsupported file type: {file_extension}")

        return result

    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        return {
            "filename": filename,
            "success": False,
            "error": str(e),
            "markdown_content": "",
            "processing_method": "failed",
        }
