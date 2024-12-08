from langchain.chains import ChatVectorDBChain
from langchain.llms import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Charger les variables d'environnement
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Vérifiez si la clé API est correctement chargée
if not openai_api_key:
    raise ValueError("Clé API OpenAI non trouvée. Vérifiez votre fichier .env.")

# Charger les embeddings OpenAI (même si inutilisés ici pour interrogation uniquement)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Charger la base de données vectorielle existante
vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Créer une instance de ChatVectorDBChain
llm = OpenAI(temperature=0.9, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)
pdf_qa = ChatVectorDBChain.from_llm(llm, vectordb, return_source_documents=True)

# Interroger la base de données
query = "Quels sont les cas d'utilisation mentionnés dans le document ?"
result = pdf_qa({"question": query, "chat_history": ""})

# Afficher la réponse
print("Answer:")
print(result["answer"])

# Afficher les documents sources (facultatif)
if "source_documents" in result:
    print("\nSource Documents:")
    for doc in result["source_documents"]:
        print(doc.page_content)
