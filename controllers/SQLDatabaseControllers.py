import json
import datetime
import httpx
import aioodbc
import numpy as np
from PIL import Image, ImageFile
from io import BytesIO
from fastapi import Request
from .BaseControllers import BaseControllers
from .FeatureExtractionControllers import FeatureExtractionControllers
from qdrant_client.http.models import PointStruct


class SQLDatabaseControllers(BaseControllers):
    def __init__(self, request: Request, is_collections_recreated: bool):
        super().__init__()
        self.request = request

        # Load last sync time
        with open("last_sync_time.json", 'r') as f:
            last_sync_data = json.load(f)

        last_sync_str = last_sync_data.get("last_sync_time", datetime.datetime.min.isoformat())
        self.last_sync_time = datetime.datetime.fromisoformat(last_sync_str)

        if is_collections_recreated:
            self.last_sync_time = datetime.datetime.min

        self.feature_extractor = FeatureExtractionControllers(request=request)
        ImageFile.LOAD_TRUNCATED_IMAGES = True


    async def get_new_rows_query(self):
        query = """
        SELECT Id, PictureUrl, ColorId, Description
        FROM Furniture
        WHERE CreatedAt > ? OR UpdatedAt > ?
        """
        async with self.request.app.SQLDatabasePool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(query, (self.last_sync_time, self.last_sync_time))
                rows = await cursor.fetchall()
                columns = [column[0] for column in cursor.description]
                return [dict(zip(columns, row)) for row in rows]

    async def get_furniture_color(self, color_id: str):
        query = """
        SELECT Name
        FROM Colors
        WHERE Id = ?
        """
        async with self.request.app.SQLDatabasePool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(query, (color_id,))
                row = await cursor.fetchone()
                if row:
                    columns = [column[0] for column in cursor.description]
                    result = dict(zip(columns, row))
                    return result["Name"]
                return "Unknown"

    async def get_img_from_url(self, img_url: str):
        if not img_url or img_url.strip() == "":
            print("Empty or invalid image URL provided.")
            return None

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(img_url)
                response.raise_for_status()

            if "image" not in response.headers.get("Content-Type", ""):
                print(f"Not an image: {img_url}")
                return None

            return Image.open(BytesIO(response.content)).convert("RGB")

        except Exception as e:
            print(f"Error fetching image: {e}")
            return None

    def extract_image_features(self, img: Image.Image):
        return self.feature_extractor.extract_image_features(
            image=img,
            model=self.request.app.model,
            processor=self.request.app.processor
        )

    def extract_text_features(self, description: str):
        return self.feature_extractor.extract_text_features(
            description=description,
            model=self.request.app.model,
            processor=self.request.app.processor
        )

    def batch_upsert(self, client, collection_name, points, batch_size=30):
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            client.upsert(collection_name=collection_name, points=batch)

    async def sync_data(self):
        rows = await self.get_new_rows_query()
        if not rows:
            return None

        img_points = []
        txt_points = []

        for row in rows:
            img = await self.get_img_from_url(row['PictureUrl'])
            if not img:
                continue

            color = await self.get_furniture_color(row['ColorId'])

            image_vector = self.extract_image_features(img)
            image_vector = np.array(image_vector, dtype=np.float32).squeeze(axis=0)

            img_points.append(PointStruct(
                id=str(row['Id']),
                vector=image_vector,
                payload={"color": color}
            ))

            text_vector = self.extract_text_features(row['Description'])
            text_vector = np.array(text_vector, dtype=np.float32).squeeze(axis=0)

            txt_points.append(PointStruct(
                id=str(row['Id']),
                vector=text_vector
            ))

        self.batch_upsert(self.request.app.client, "text_features", txt_points)
        self.batch_upsert(self.request.app.client, "image_features", img_points)

        with open("last_sync_time.json", 'w') as f:
            json.dump({
                "last_sync_time": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)
