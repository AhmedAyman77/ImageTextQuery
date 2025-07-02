from helpers.config import get_settings_object, Settings

class BaseControllers:
    def __init__(self):
        self.app_settings = get_settings_object()
    