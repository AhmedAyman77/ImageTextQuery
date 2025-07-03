import asyncio
from fastapi import APIRouter, Request
from qdrant_client.http import models
from controllers import SQLDatabaseControllers
from apscheduler.schedulers.background import BackgroundScheduler

sync_data_router = APIRouter(
    prefix="/api/v1/sync",
    tags=["api_v1", "update"],
)

scheduler = BackgroundScheduler()
scheduler_started = False


@sync_data_router.post("/RecreateCollection")
async def recreate_collections(request: Request):
    # Recreate Qdrant collections
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

    # Create controller and sync
    await SQLDatabaseControllers(
        request=request,
        is_collections_recreated=True
    ).sync_data()

    return {"status": "Collections Recreated"}




@sync_data_router.post("/SyncData")
async def manual_update(request: Request):

    await SQLDatabaseControllers(
        request=request,
        is_collections_recreated=False
    ).sync_data()

    return {"status": "Triggered sync from API"}




# Wrap async sync in a thread-safe job for scheduler
def sync_job_wrapper(request: Request):
    loop = asyncio.get_event_loop()
    loop.create_task(run_sync(request))

async def run_sync(request: Request):
    obj = SQLDatabaseControllers(
        request=request,
        is_collections_recreated=False
    )
    await obj.sync_data()


@sync_data_router.post("/start_scheduler")
async def start_scheduler(request: Request):
    global scheduler_started
    if scheduler_started:
        return {"status": "Scheduler already running"}

    scheduler.add_job(sync_job_wrapper, 'interval', minutes=10, args=[request])
    scheduler.start()
    scheduler_started = True

    return {"status": "Scheduler started (every 10 minutes)"}