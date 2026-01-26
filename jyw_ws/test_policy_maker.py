from typing import TypeVar
from torch import nn

T = TypeVar("T", bound="PreTrainedPolicy")

class PreTrainedPolicy(nn.Module):
    @classmethod
    def from_pretrained(cls:T) -> T:
        print("cls is:", cls)
        return cls()   # ← 用“类”构造实例

class MyPolicy(PreTrainedPolicy):
    pass

p = MyPolicy.from_pretrained()
print(p,type(p))
