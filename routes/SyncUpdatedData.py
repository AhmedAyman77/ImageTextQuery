from fastapi import APIRouter, Request
from qdrant_client.http import models
from controllers import SQLDatabaseControllers
from apscheduler.schedulers.background import BackgroundScheduler

sync_data_router = APIRouter(
    prefix="/api/v1/sync",
    tags=["api_v1", "update"],
)


@sync_data_router.post("/RecreateCollection")
def recreate_collections(request: Request):
    request.app.client.recreate_collection(
        collection_name="image_features",
        vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),
    )
    request.app.client.create_payload_index(
        collection_name="image_features",
        field_name="color",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
    request.app.client.recreate_collection(
        collection_name="text_features",
        vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),
    )

    SQLDatabaseControllers(
        request=request,
        is_collections_recreated=True
    ).sync_data()

    return {
        "status": "Collections Recreated",
    }


@sync_data_router.post("/SyncData")
def manual_update(request: Request):
    SQLDatabaseControllers(
        request=request,
        is_collections_recreated=False
    ).sync_data()

    return {
        "status": "Triggered sync from API"
    }


@sync_data_router.post("/start_scheduler")
def Database_scheduler(request: Request):
    # Scheduler to run every 10 minutes
    scheduler = BackgroundScheduler()
    sync_updated_furniture = SQLDatabaseControllers(
        request=request, 
        is_collections_recreated=False
    ).sync_data()
    scheduler.add_job(sync_updated_furniture, 'interval', minutes=10)
    scheduler.start()
    