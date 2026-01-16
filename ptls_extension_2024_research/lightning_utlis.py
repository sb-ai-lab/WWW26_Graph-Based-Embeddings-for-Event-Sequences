from typing import Callable
from dataclasses import dataclass, field

@dataclass
class LogLstEl:
    name: str
    value: float
    args: list = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)

    def alter_name(self, name_alterer: Callable):
        return LogLstEl(name = name_alterer(self.name), 
                        value = self.value,
                        args = self.args,
                        kwargs = self.kwargs)
    
    def alter_name_(self, name_alterer: Callable):
        self.name = name_alterer(self.name)
        return self