import json
import datetime
import requests
import numpy as np
from PIL import Image, ImageFile
from io import BytesIO
from fastapi import Request
from .BaseControllers import BaseControllers
from .FeatureExtractionControllers import FeatureExtractionControllers
from qdrant_client.http.models import PointStruct

class SQLDatabaseControllers(BaseControllers):
    def __init__(
        self,
        request: Request,
        is_collections_recreated: bool
    ):
        super().__init__()
        self.request = request

        with open("last_sync_time.json", 'r') as f:
            last_sync_data = json.load(f)

        # Convert string to datetime
        last_sync_str = last_sync_data.get("last_sync_time", datetime.datetime.min.isoformat())
        self.last_sync_time = datetime.datetime.fromisoformat(last_sync_str)

        # get the time {time_in_minutes} ago
        if is_collections_recreated:
            self.last_sync_time = datetime.datetime.min
        

        self.cursor = request.app.SQLDatabaseClient.cursor(as_dict=True)
        
        self.feature_extractor = FeatureExtractionControllers(request=request)

        ImageFile.LOAD_TRUNCATED_IMAGES = True  # load the broken image anyway


    def get_new_rows_query(self):
        query = f"""
        SELECT * FROM Furniture
        WHERE CreatedAt > %s OR UpdatedAt > %s
        """
        self.cursor.execute(query, (self.last_sync_time, self.last_sync_time))
        rows = self.cursor.fetchall()
        
        return rows
    
    def get_furniture_color(
        self,
        color_id: str
    ):
        query = "SELECT Name FROM Colors WHERE Id = %s"
        self.cursor.execute(query, (color_id,))
        result = self.cursor.fetchone()
        color = result["Name"]

        return color
    
    def get_img_from_url(self, img_url: str):
        if not img_url or img_url.strip() == "":
            print("Empty or invalid image URL provided.")
            return None
        
        response = requests.get(img_url)
        response.raise_for_status()

        if "image" not in response.headers.get("Content-Type", ""):
            print(f"Not an image: {img_url}")
            return None
        
        try:
            img = Image.open(BytesIO(response.content)).convert("RGB")
            return img
    
        except Exception as e:
            raise ValueError(f"Could not open image from URL: {img_url}") from e
    
    def extract_image_features(self, img: Image.Image):
        image_vector = self.feature_extractor.extract_image_features(
            image=img,
            model=self.request.app.model,
            processor=self.request.app.processor
        )

        return image_vector

    def extract_text_features(self, description: str):
        text_vector = self.feature_extractor.extract_text_features(
            description=description,
            model=self.request.app.model,
            processor=self.request.app.processor
        )

        return text_vector
    
    def batch_upsert(self, client, collection_name, points, batch_size=30):
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            client.upsert(collection_name=collection_name, points=batch)

    def sync_data(self):
        # get all updated furniture items since last_sync_time
        rows = self.get_new_rows_query()

        if rows:
            img_points = []
            txt_points = []
            
            for row in rows:
                # get image from url
                img_url = row['PictureUrl']
                img = self.get_img_from_url(
                    img_url=img_url
                )
                
                if not img:
                    continue
                
                # get furniture color
                color = self.get_furniture_color(
                    color_id=row['ColorId']
                )
                
                # Extract image features
                image_vector = self.extract_image_features(img=img)
                image_vector = np.array(image_vector).astype(np.float32)
                image_vector = image_vector.squeeze(axis=0)


                # prepare img features points
                img_points.append(
                    PointStruct(
                        id=str(row['Id']),
                        vector=image_vector,
                        payload={
                            "color": color,
                        }
                    )
                )
                

                # Extract text features
                description = row['Description']
                text_vector = self.extract_text_features(description=description)
                text_vector = np.array(text_vector).astype(np.float32)
                text_vector = text_vector.squeeze(axis=0)
                

                # prepare text features points
                txt_points.append(
                    PointStruct(
                        id=str(row['Id']),
                        vector=text_vector
                    )
                )
            
            # index the features in qdrant in batches
            self.batch_upsert(self.request.app.client, "text_features", txt_points, batch_size=30)
            self.batch_upsert(self.request.app.client, "image_features", img_points, batch_size=30)

            # save the current sync time to last_sync_time.json
            with open("last_sync_time.json", 'w') as f:
                json.dump({
                    "last_sync_time": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2)
