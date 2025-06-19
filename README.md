---
title: ImageTextQuery
emoji: 🌖
colorFrom: purple
colorTo: gray
sdk: docker
pinned: false
license: apache-2.0
---

# ImageTextQuery
A multimodal search engine that allows users to search for relevant images using natural language text or image queries. By combining the power of computer vision, NLP embeddings, and vector similarity search (Qdrant), this engine enables intuitive and semantic retrieval of visual content.

# Technologies used
- **FastAPI** for building the RESTful API

- **Qdrant** for store and retrieve image and text embeddings efficiently based on semantic similarity

- **NLP Embeddings** Used to convert natural language text queries into high-dimensional vectors for semantic comparison with image embeddings.

- **Computer Vision Embeddings**
Used to extract feature vectors from images so they can be compared with text or other images in the vector space.

# getting started

1- ***Clone the repo***   
```bash
git clone git@github.com:AhmedAyman77/ImageTextQuery.git
cd ImageTextQuery
```

2- ***create conda env***
```bash
conda create -n Image-Text-Query python=3.11
```

3- ***Activate the environment***
```bash
conda activate Image-Text-Query
```

4- ***Install the required packages***
```bash
pip install -r requirements.txt
```

5- ***run FastAPI server***
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 5000
```

### test Endpoints
- you can use this endpoint for search using image  
**http://0.0.0.0:5000/search/image**

- or use this endpoint for search by writing a description about the furniture you want **http://0.0.0.0:5000/search/text**
