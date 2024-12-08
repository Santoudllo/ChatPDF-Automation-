from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

# Définir le chemin relatif
base_path = Path(__file__).resolve().parent.parent / "docs"
pdf_path = base_path / "exemple_test.pdf"

# Charger le PDF et diviser en pages
loader = PyPDFLoader(str(pdf_path))
pages = loader.load_and_split()

# Afficher le contenu de la première page
print(pages[0].page_content)
