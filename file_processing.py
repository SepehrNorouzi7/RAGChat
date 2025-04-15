import fitz
import docx
from io import BytesIO

class FileHandler:
    def extract_text(self, uploaded_file):
        """
        Extract text from PDF or DOCX files.
        """
        try:
            if uploaded_file.name.endswith(".pdf"):
                return self._extract_pdf(uploaded_file)
            elif uploaded_file.name.endswith(".docx"):
                return self._extract_docx(uploaded_file)
            else:
                raise ValueError("Unsupported file format")
        except Exception as e:
            raise Exception(f"Error processing {uploaded_file.name}: {str(e)}")
    
    def _extract_pdf(self, uploaded_file):
        text = ""
        # Use PyMuPDF to open and read the PDF from a stream
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text

    def _extract_docx(self, uploaded_file):
        # Use python-docx to extract text from a DOCX file
        doc = docx.Document(BytesIO(uploaded_file.read()))
        return "\n".join([p.text for p in doc.paragraphs])
