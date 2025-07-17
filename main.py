import pathway as pw
import re
import numpy as np
from pathway_tools import extract_pdf_text, clean_and_split, embed_sentences
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# View file content
table = pw.io.gdrive.read(
    object_id="1dlzAYEqKrlJqYkuVRkandoyYfPA7bE1V",
    service_user_credentials_file="credentials.json",
    mode="static"
)

class MySchema(pw.Schema):
    data: bytes
table = table._with_schema(MySchema)

text_table = table.select(text=extract_pdf_text(table.data))    
sentence_table = text_table.select(sentences=clean_and_split(text_table.text))
vector_table = sentence_table.select(
    embeddings=embed_sentences(sentence_table.sentences)
)

df = pw.debug.table_to_pandas(vector_table)
embeddings = np.vstack([np.array(e).squeeze() for e in df["embeddings"]])

cosine_matrix = cosine_similarity(embeddings)
n = len(embeddings)
upper_tri_indices = np.triu_indices(n, k=1)
avg_score = cosine_matrix[upper_tri_indices].mean()
std_score = cosine_matrix[upper_tri_indices].std()
print(f"Average cosine similarity: {avg_score}, Std Dev: {std_score}")