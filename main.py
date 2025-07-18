import pathway as pw
import re
import numpy as np
import hdbscan
from pathway_tools import extract_pdf_text, clean_and_split, embed_sentences
from sklearn.ensemble import RandomForestClassifier
from ml_functions import extract_hdbscan_features
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


files = [
    {"fileId": "1BwVMU26RRu6a8fZ0qmS8leUeXQHjIshD"},
    {"fileId": "1kXXdZw8hZrBwQ3oANma0DIlKE7Usw1Q9"},
    {"fileId": "1PKYuPUXD2q39WGgUGvfl8KvUHbZzEV7i"},
    {"fileId": "160QUt4yDHSV0NSByx66Gr_cQ5iHv8way"},
    {"fileId": "1akkfw4dLYyOdezjQFG3WsnFllKa5VVaa"},
    {"fileId": "1YjVysxcpVpDwQnXfQ8V4iH7-WLA7ct1C"},
    {"fileId": "14kGRS5NlTyWfO1Pz0zQqu7MMjKgimG22"},
    {"fileId": "1BukEjc3bT8mHTD-TCxjVMzs452qEFAnX"},
    {"fileId": "1jOLE3bWXWIQZpGL_KnpZlXRVdLdvxMQi"},
    {"fileId": "1hbzBB8u9aGQqwjPrVEL20nbNa7aNyggX"},
    {"fileId": "1JvyOi5cFWkSxiEpsVTvdY82DyGMBOOZi"},
    {"fileId": "1AE5Q619M52Gb3LvKTOJNv6bEjmrkLo0t"},
    {"fileId": "1qNFPCdxQaoMy1wA0D8Dq0pU2qc1qiyGt"},
    {"fileId": "1ldiYOjVY517vxrzPHaMdtyQmNzGLZEG6"},
    {"fileId": "16kCZqG7KrvRqoJ5q5u5GtQJzfQzfDfP4"}
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

        text_table = table.select(text=extract_pdf_text(table.data))    
        sentence_table = text_table.select(sentences=clean_and_split(text_table.text))
        vector_table = sentence_table.select(
            embeddings=embed_sentences(sentence_table.sentences)
        )

        df = pw.debug.table_to_pandas(vector_table)
        embeddings = np.vstack([np.array(e).squeeze() for e in df["embeddings"]])

        feature = extract_hdbscan_features(embeddings)
        features.append(feature)
    except Exception as e:
        print(f"Error processing {file_id}: {str(e)}")
        features.append(None)

pw.run()

X = np.array(features)
Y = np.array(labels)

model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X, Y)

test = np.array([8, 0.1, 0.05]).reshape(1, -1)  # Example test input
prediction = model.predict(test)
print(f"Prediction for test input {test}: {prediction}")

feature_names = ["n_clusters", "largest_cluster_ratio", "avg_outlier_score"]

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