from fastapi import Request

class SearchControllers:
    
    def __init__(
        self,
        limit: int,
        request: Request
    ):
        super().__init__()
        self.limit = limit
        self.request = request
        self.client = self.request.app.client

    def get_cursor(self):
        return self.request.app.SQLDatabaseClient.cursor(as_dict=True)
    
    def url_response(self, search_res):
        cursor = self.get_cursor()
        urls_response = []
        for hit in search_res:
            query = "SELECT Id, PictureUrl FROM Furniture WHERE Id = %s"
            cursor.execute(query, (hit.id,))
            result = cursor.fetchone()
            result["Id"] = str(result["Id"])
            urls_response.append(result)
        
        return urls_response
