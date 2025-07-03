from fastapi import Request

class SearchControllers:
    
    def __init__(self, limit: int, request: Request):
        self.limit = limit
        self.request = request
        self.client = self.request.app.client

    async def url_response(self, search_res):
        urls_response = []

        async with self.request.app.SQLDatabasePool.acquire() as conn:
            async with conn.cursor() as cursor:
                for hit in search_res:
                    query = """
                    SELECT Id, PictureUrl
                    FROM Furniture
                    WHERE Id = ?
                    """
                    await cursor.execute(query, (hit.id,))
                    row = await cursor.fetchone()

                    if row:
                        # Convert tuple row to dict using cursor.description
                        columns = [col[0] for col in cursor.description]
                        result = dict(zip(columns, row))

                        result["Id"] = str(result["Id"])
                        urls_response.append(result)

        return urls_response
