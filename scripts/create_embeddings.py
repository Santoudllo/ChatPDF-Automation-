from dotenv import load_dotenv
import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Charger la clé API OpenAI depuis les variables d'environnement
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("La clé API OpenAI n'est pas configurée. Assurez-vous qu'elle est dans le fichier .env")

# Créer les embeddings avec OpenAI
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Charger un fichier PDF
pdf_path = "/home/santoudllo/Bureau/PROJETS/ChatPDF-Automation-/docs/exemple_test.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()

# Créer une base de données vectorielle avec Chroma
vectordb = Chroma.from_documents(pages, embedding=embeddings, persist_directory="./chroma_db")

# Sauvegarder la base de données
vectordb.persist()

print("Base de données vectorielle créée et sauvegardée avec succès.")
