from dataclasses import dataclass,field
from typing import Optional

import parser

# Define your configuration as a dataclass
@dataclass(slots=True)
class TrainPipelineConfig:
    width: int = 640
    height: int = 480
    raw_fps: int = 30
    step: int = 2
    robot: str = "agx"
    task_text: str = "fold the blue cloth towel neatly"
    data_dir: str = "/workspace/data/downloads/537e9a6f62034a19aad056fe21b53fbb"
    workers: int = 16
    episodes_per_dataset: int = 5
    batch_encoding_size: int = 0
    image_writer_threads: int = 0
    image_writer_processes: int = 0
    parallel_encoding: bool = False
    head_keep: int = 5
    tail_keep: int = 1
    eps: float = 1e-2
    force_start0: bool = False
    note: str = ""
    fps: int = field(init=False)  # 声明为字段，但不从外部传入

    def __post_init__(self):
        self.fps = self.raw_fps // self.step



# Example function using the @parser.wrap() decorator
@parser.wrap()
def train(cfg: TrainPipelineConfig):
    # cfg.process()
    print(f"Training with configuration: {cfg}")
    print(cfg.fps)

# if __name__ == "__main__":
train()
