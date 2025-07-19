import pathway as pw
import re
import numpy as np
import hdbscan
from pathway_tools import extract_pdf_text, clean_and_split, embed_sentences
from sklearn.ensemble import RandomForestClassifier
from ml_functions import extract_hdbscan_features, compute_readability_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials

def get_folder_files(folder_id="1t3bRu8MyW3GC-DvMYvZ5ckDG60_cMHzc"):
    """Fetch all PDF file IDs from a given Google Drive folder, even if more than 100."""

    credentials = Credentials.from_service_account_file(
        "credentials.json",
        scopes=['https://www.googleapis.com/auth/drive.readonly']
    )

    service = build('drive', 'v3', credentials=credentials)

    all_files = []
    page_token = None

    while True:
        response = service.files().list(
            q=f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false",
            fields="nextPageToken, files(id, name)",
            pageSize=100,
            pageToken=page_token
        ).execute()

        files = response.get('files', [])
        all_files.extend(files)

        page_token = response.get('nextPageToken')
        if not page_token:
            break  # No more pages

    print(f"Found {len(all_files)} PDF file(s):")

    return [{"fileId": file['id'], "name": file['name']} for file in all_files]

# Run it
training_files = get_folder_files()

files = [
    {"fileId": "1BwVMU26RRu6a8fZ0qmS8leUeXQHjIshD", "label": 0},
    {"fileId": "1kXXdZw8hZrBwQ3oANma0DIlKE7Usw1Q9", "label": 1},
    {"fileId": "1PKYuPUXD2q39WGgUGvfl8KvUHbZzEV7i", "label": 1},
    {"fileId": "160QUt4yDHSV0NSByx66Gr_cQ5iHv8way", "label": 0},
    {"fileId": "1akkfw4dLYyOdezjQFG3WsnFllKa5VVaa", "label": 1},
    {"fileId": "1YjVysxcpVpDwQnXfQ8V4iH7-WLA7ct1C", "label": 1},
    {"fileId": "14kGRS5NlTyWfO1Pz0zQqu7MMjKgimG22", "label": 1},
    {"fileId": "1BukEjc3bT8mHTD-TCxjVMzs452qEFAnX", "label": 0},
    {"fileId": "1jOLE3bWXWIQZpGL_KnpZlXRVdLdvxMQi", "label": 0},
    {"fileId": "1hbzBB8u9aGQqwjPrVEL20nbNa7aNyggX", "label": 1},
    {"fileId": "1JvyOi5cFWkSxiEpsVTvdY82DyGMBOOZi", "label": 1},
    {"fileId": "1AE5Q619M52Gb3LvKTOJNv6bEjmrkLo0t", "label": 1},
    {"fileId": "1qNFPCdxQaoMy1wA0D8Dq0pU2qc1qiyGt", "label": 0},
    {"fileId": "1ldiYOjVY517vxrzPHaMdtyQmNzGLZEG6", "label": 1},
    {"fileId": "16kCZqG7KrvRqoJ5q5u5GtQJzfQzfDfP4", "label": 1},
]

labels = [0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1]

features = []

# View file content
for file in files:
    file_id = file["fileId"]

    try:
        table = pw.io.gdrive.read(
            object_id=file_id,
            service_user_credentials_file="credentials.json",
            mode="static"
        )

        class MySchema(pw.Schema):
            data: bytes
        table = table._with_schema(MySchema)

        # Extract text
        text_table = table.select(text=extract_pdf_text(table.data))
        full_text = pw.debug.table_to_pandas(text_table)["text"][0]

        # Compute readability
        readability_score = compute_readability_score(full_text)

        # Sentence embeddings + HDBSCAN
        sentence_table = text_table.select(sentences=clean_and_split(text_table.text))
        vector_table = sentence_table.select(embeddings=embed_sentences(sentence_table.sentences))
        df = pw.debug.table_to_pandas(vector_table)
        embeddings = np.vstack([np.array(e).squeeze() for e in df["embeddings"]])
        hdbscan_features = extract_hdbscan_features(embeddings)

        # Combine all features
        combined = hdbscan_features + [readability_score]
        features.append(combined)
        print("training extraction done")
    except Exception as e:
        print(f"Error processing {file_id}: {str(e)}")
        features.append([0.0, 0.0, 0.0, 0.0])  # fallback in case of error



testing_features = []

for file in training_files:
    file_id = file["fileId"]

    try:
        table = pw.io.gdrive.read(
            object_id=file_id,
            service_user_credentials_file="credentials.json",
            mode="static"
        )

        class MySchema(pw.Schema):
            data: bytes
        table = table._with_schema(MySchema)

        # Extract text
        text_table = table.select(text=extract_pdf_text(table.data))
        full_text = pw.debug.table_to_pandas(text_table)["text"][0]

        # Compute readability
        readability_score = compute_readability_score(full_text)

        # Sentence embeddings + HDBSCAN. 
        sentence_table = text_table.select(sentences=clean_and_split(text_table.text))
        vector_table = sentence_table.select(embeddings=embed_sentences(sentence_table.sentences))
        df = pw.debug.table_to_pandas(vector_table)
        embeddings = np.vstack([np.array(e).squeeze() for e in df["embeddings"]])
        hdbscan_features = extract_hdbscan_features(embeddings)

        # Combine all features
        combined = hdbscan_features + [readability_score]
        testing_features.append(combined)

    except Exception as e:
        print(f"Error processing {file_id}: {str(e)}")
        testing_features.append([0.0, 0.0, 0.0, 0.0])  # fallback in case of error

pw.run()

print("Extracted features:", features)

X = np.array(features)
Y = np.array(labels)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, Y)

test = np.array(testing_features)
prediction = model.predict(test)
print(f"Prediction for test input {test}: {prediction}")

feature_names = ["n_clusters", "largest_cluster_ratio", "avg_outlier_score", "readability_score"]

# Create DataFrame for features and labels
df = pd.DataFrame(X, columns=feature_names)
df["label"] = Y

# Compute correlation matrix
correlation_matrix = df.corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation between Features and Label (Publishability)")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")  # Saves graph to a file
