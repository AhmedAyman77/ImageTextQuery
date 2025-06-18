from helpers.config import get_settings_object, Settings

class BaseController:
    def __init__(self):
        self.app_settings = get_settings_object()
    