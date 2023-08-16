from .action import ActionCount, action_bus
from .component import Component
from .environment import environment_bus

# TODO: try to import 'gym' and 'gymnasium' for proper typing
# TODO: optional dependency: pip install roc[gym] or roc[gymnasium]

try:
    import gym
except Exception:
    pass


class GymComponent(Component):
    def __init__(self, gym: gym) -> None:
        super().__init__("gym-interface", "environment")
        self.gym = gym
        self.env_bus = environment_bus
        self.action_bus = action_bus
        self.env_bus_conn = self.env_bus.connect(self)
        self.action_bus_conn = self.action_bus.connect(self)

    def send_actions(self, action_count: int) -> None:
        a = ActionCount(action_count=action_count)
        self.action_bus_conn.send(a)
