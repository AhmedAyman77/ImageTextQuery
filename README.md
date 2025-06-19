# ImageTextQuery
A multimodal search engine that allows users to search for relevant images using natural language text or image queries. By combining the power of computer vision, NLP embeddings, and vector similarity search (Qdrant), this engine enables intuitive and semantic retrieval of visual content.

# Technologies used
- **FastAPI** for building the RESTful API

- **Qdrant** for store and retrieve image and text embeddings efficiently based on semantic similarity

- **CLIP** Used to convert natural language text queries or image query into high-dimensional vectors.


# APIs
### Search for relevant images by providing a natural language description.   
### 🔸 Using Postman:    
- Set method: POST    
- Set URL: https://A7medAyman-image-text-search.hf.space/api/v1/text/TextSearchQuery

 - Go to the Body tab → choose form-data

 - Add field:
    - **description** with your search query (required)
    - **limit** with number of images you want (optional)

### Find similar images by uploading a reference image.   
### 🔸 Using Postman:
- Set method: POST  

- Set URL: https://A7medAyman-image-text-search.hf.space/api/v1/image/ImageSearchQuery

 - Go to the Body tab → choose form-data

 - Add field:

    - **File** upload an image file (required)
    - **limit** number of images to return (optional)
    - **color** filter by color (optional, e.g., "black")