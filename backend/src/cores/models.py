from typing import TypedDict

class VideoModel(TypedDict):
    id: str
    path: str

class ShotModel(TypedDict):
    id: str
    start: int
    end: int
    videoId: str

class ObjectFeatureModel(TypedDict):
    label: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    score: float

class FrameModel(TypedDict):
    id: str
    shotId: str
    selected: bool
    objects: list[ObjectFeatureModel]
    path: str
