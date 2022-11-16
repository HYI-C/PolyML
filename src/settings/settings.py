class Borg:
    __shared_state = {}

    def __init__(self):
        self.__dict__ = self.__shared_state

class Settings(Borg):
    def configure(
        self,
        quickstart_settings = "custom",
    ):
        self._configure_settings(quickstart_settings)

        return

    def _configure_settings(self, quickstart_settings: str):
        if quickstart_settings == "low":
            import settings.quickstart1 as settings_module
        if quickstart_settings == "high":
            import settings.quickstart2 as settings_module
        if quickstart_settings == "default":
            import settings.quickstart as settings_module
        else:
            import settings.custom as settings_module


        for var in dir(settings_module):
            if not var.startswith("__"):
                self.__dict__[var] = getattr(settings_module, var)
        
        return
        