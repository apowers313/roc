from pydantic import BaseModel

from .event import EventBus


# TODO: action space config
# TODO: vision input
# TODO: sound input
# TODO: other input
class EnvData(BaseModel):
    pass


environment_bus = EventBus[EnvData]("environment")
