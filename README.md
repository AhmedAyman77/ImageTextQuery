# ImageTextQuery: Multimodal Semantic Image Search

**ImageTextQuery** is an advanced multimodal search engine that revolutionizes the way users interact with visual content. Combining state-of-the-art computer vision and natural language processing, ImageTextQuery enables users to search for relevant images using either natural language queries or by uploading a reference image.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [File Structure](#file-structure)
- [Controllers Directory Overview](#controllers-directory-overview)
  - [BaseControllers.py](#1-basecontrollerspy)
  - [FeatureExtractionControllers.py](#2-featureextractioncontrollerspy)
  - [SQLDatabaseControllers.py](#3-sqldatabasecontrollerspy)
  - [Other Controllers (SearchControllers, SemanticSearchControllers, SimilaritySearchControllers)](#other-controllers)
- [Routes Directory Overview](#routes-directory-overview)
  - [ImageSearch.py](#1-imagesearchpy)
  - [TextSearch.py](#2-textsearchpy)
  - [SyncUpdatedData.py](#3-syncupdateddatapy)
- [Get Started](#get-started)
  - [Clone the Repository](#1-clone-the-repository)
  - [Set Up Your Python Environment](#2-set-up-your-python-environment)
    - [Using virtualenv](#option-1-using-virtualenv)
    - [Using conda](#option-2-using-conda)
  - [Download the CLIP Model](#3-download-the-clip-model)
  - [Configure Environment Variables](#4-configure-environment-variables)
  - [Run the Application](#5-run-the-application)
  - [Using the API](#6-using-the-api)
- [Contact](#contact)

---

## How It Works

- **Multimodal Embeddings:** User queries—whether text or images—are converted into high-dimensional vectors using the powerful CLIP model.
- **Vector Database:** These embeddings are stored and indexed in Qdrant, a high-performance vector database optimized for semantic similarity search.
- **Semantic Retrieval:** When a query is made, its embedding is compared with stored vectors, retrieving images whose content or description most closely matches the user's request.
- **Flexible Input:** Users can search by:
  - Describing what they want in natural language (e.g., “A black armchair with wooden legs”)
  - Providing a sample/reference image to find visually similar items

## Key Features

- **Natural Language Search:** Find images by describing them in your own words.
- **Image-Based Search:** Upload an image to discover visually or semantically similar images.
- **Color and Attribute Filtering:** Optionally filter results by specific colors or other attributes.
- **Fast, Scalable API:** Powered by FastAPI for efficient and scalable RESTful endpoints.

## Technologies Used

- **CLIP (Contrastive Language-Image Pretraining):** Converts text and images into comparable feature vectors.
- **Qdrant:** Stores and retrieves embeddings efficiently based on semantic similarity.
- **FastAPI:** Provides a robust, modern API interface.
- **spaCy:** Powers advanced text preprocessing and lemmatization.

---

## File Structure
```
ImageTextQuery/
│
├── controllers/
│   ├── BaseControllers.py
│   ├── FeatureExtractionControllers.py
│   └── SQLDatabaseControllers.py
│   └── SearchControllers.py
│   └── SemanticSearchControllers.py
│   └── SimilaritySearchControllers.py
│
├── helpers/
│   └── config.py
│
├── routes/
│   ├── ImageSearch.py
│   ├── TextSearch.py
│   └── SyncUpdatedData.py
│
├── notebook/
│   └── local_TextSearch.ipynb
│
├── main.py
├── requirements.txt
├── Dockerfile    -----> for hugging face deploy
├── README.md
└── last_sync_time.json
```

---

## Controllers Directory Overview

The `controllers` directory contains the core backend logic for handling data processing, feature extraction, and database synchronization in the ImageTextQuery project. Here’s a detailed description of each file:

### 1. BaseControllers.py
**Purpose:**  
Defines a foundational base class for controllers in the project. Common logic, shared methods, utility functions, and standard behaviors for all controllers are placed here, ensuring that code is reusable and maintainable across different modules.

**Key Roles:**
- Acts as the parent class for other controllers.
- Provides shared infrastructure.

### 2. FeatureExtractionControllers.py
**Purpose:**  
Handles the extraction and processing of features from both text and image data. This includes text cleaning, lemmatization, and conversion of inputs to embeddings using the CLIP model.

**Key Roles:**
- Cleans and preprocesses input text.
- Converts text descriptions to embeddings (vectors) via NLP and the CLIP model.
- Processes images to extract high-dimensional feature vectors.
- Prepares data for semantic similarity search.

### 3. SQLDatabaseControllers.py
**Purpose:**  
Manages interactions with the SQL database and coordinates the synchronization of structured data with the vector database (Qdrant). Responsible for fetching new/updated records, extracting features, and updating the vector store.

**Key Roles:**
- Connects to and queries the SQL database.
- Downloads and processes images from URLs in the database.
- Extracts and stores both image and text features as embeddings.
- Performs batch upsert operations to Qdrant collections for both modalities.
- Maintains synchronization state to ensure data freshness.

---

## Routes Directory Overview

The `routes` directory contains the API route definitions for the ImageTextQuery project. Each file defines a set of RESTful endpoints that expose the core functionality of the application through FastAPI routers.

### 1. ImageSearch.py
**Purpose:**  
Defines API endpoints for image-based semantic search.

**Key Roles:**
- Accepts image file uploads as queries.
- Extracts features from the uploaded image using the CLIP model.
- Searches for and returns images that are visually or semantically similar to the query image.
- Supports additional filters (e.g., color).

### 2. TextSearch.py
**Purpose:**  
Defines API endpoints for text-based semantic search.

**Key Roles:**
- Accepts text descriptions as search queries.
- Processes and converts text queries into embeddings using NLP and the CLIP model.
- Retrieves and returns images from the database that are semantically similar to the provided text query.

### 3. SyncUpdatedData.py
**Purpose:**  
Handles data synchronization between the SQL database and the vector database (Qdrant).

**Key Roles:**
- Provides endpoints to trigger or manage the syncing of new or updated data.
- Ensures that the latest images and descriptions are indexed and available for search.
- Supports maintaining data freshness and consistency in the search index.

---

## Get Started

Follow these steps to set up and run the ImageTextQuery project on your local machine:

### 1. Clone the Repository

```bash
git clone https://github.com/AhmedAyman77/ImageTextQuery.git
cd ImageTextQuery
```

### 2. Set Up Your Python Environment

You can use either **virtualenv** or **conda** to manage your environment.

#### Option 1: Using virtualenv

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Option 2: Using conda

```bash
conda create -n imagetextquery python=3.11
conda activate imagetextquery
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Download the CLIP Model

Use the Hugging Face CLI or manually download the files using:

```python
from transformers import CLIPProcessor, CLIPModel

CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
```

### 4. Configure Environment Variables

Create a `.env` file or set the required environment variables for:

- Qdrant host and API key
- SQL server connection details

Example `.env` contents:

```
QDRANT_HOST=http://localhost:6333
QDRANT_API_KEY=your-qdrant-api-key
SQL_HOST=localhost
SQL_PORT=1433
SQL_DATABASE=your_database
SQL_USER=your_username
SQL_PASSWORD=your_password
```

### 5. Run the Application

You can start the FastAPI server using Uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 7860
```

The API will be available at:  
[http://localhost:7860/docs](http://localhost:7860/docs) (for Swagger UI)

### 6. Using the API

- For **text-based search**, use the `/api/v1/text/TextSearchQuery` endpoint.
- For **image-based search**, use the `/api/v1/image/ImageSearchQuery` endpoint.

You can interact with the API using tools like [Postman](https://www.postman.com/) or cURL.

---

## Contact

For questions, suggestions, or collaborations, please contact:

- **Ahmed Ayman**
- Email: [devahmedaymn@gmail.com](mailto:devahmedaymn@gmail.com)
- LinkIn: [LinkIn Profile](https://www.linkedin.com/in/ahmed-ayman-25a9b2248/)