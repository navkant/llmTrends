from sympy import principal_branch
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import os
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


# Configure embedding model to be used by the database to generate embedding
model_name = "ViT-B-32"
embedding_function = OpenCLIPEmbeddingFunction(model_name=model_name)

# Loads images from image URI's given to the database
data_loader = ImageLoader()

# Create chroma client
client = chromadb.Client()
collection_name="multimodal_embeddings_collection"


# Create chroma collection
collection = client.create_collection(
    name=collection_name,
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"},
    data_loader=data_loader)


# ****Prepare dataset to be added to the collection****
# Load the CSV file with image IDs and descriptions
csv_file = 'image_descriptions.csv'
df = pd.read_csv(csv_file)

# Folder containing the images
images_folder = 'images/out'
# Prepare lists for image paths, image ids, descriptions, and description ids
image_paths = []
image_Ids = []
descriptions = []
description_Ids = []

# Iterate through the CSV file to get image paths and corresponding descriptions
for _, row in df.iterrows():
    des_id = str(row['Image ID'])  # Ensure the ID is a string for matching
    description_Ids.append(des_id)
    description = str(row['Description'])
    # Find the image file corresponding to the image_id
    for file_name in os.listdir(images_folder):
        if file_name.startswith(f"{des_id}_") and file_name.endswith('.png'):
            image_path = os.path.join(images_folder, file_name)
            image_Ids.append(file_name)
            image_paths.append(image_path)
            descriptions.append(description)
            break

# Add images and descriptions to the collection
for img_id, img_path, desc_id, desc in zip(image_Ids, image_paths, description_Ids, descriptions):
    collection.add(
        ids=[img_id],
        uris=[img_path],
        metadatas=[{"image_uri": img_path, "description": desc}]
    )
    collection.add(
        ids=[desc_id],
        documents=[desc],
        metadatas=[{"image_uri": img_path, "description": desc}]
    )

# Query by text
query_text= "vitamic C fruits"
text_query_results = collection.query(
    query_texts=[query_text],
    n_results=5
)

for metadata, distance in zip(text_query_results['metadatas'][0], text_query_results['distances'][0]):
    print(f"Description: {metadata['description']}")
    print(f"Distance: {distance:.4f}")
    # display(IPImage(filename=metadata['image_uri']))


# Query by image
query_images=[]
query_image_path = '/usr/local/datasetsDir/images-and-descriptions/queries/girlwithorangesliceoneyes.jpg'
query_images.append(query_image_path)
image_query_results = collection.query(
    query_uris=query_images,
    n_results=2
)
print("Image query: ")
# display(IPImage(filename=query_image_path))
print("Image query results array returned by the database: ", image_query_results)
print("*********** Visualizing the results ************")
for metadata, distance in zip(image_query_results['metadatas'][0], image_query_results['distances'][0]):
    print(f"Description: {metadata['description']}")
    print(f"Distance: {distance:.4f}")
    # display(IPImage(filename=metadata['image_uri']))

# Delete the collection
client.delete_collection(name=collection_name)
