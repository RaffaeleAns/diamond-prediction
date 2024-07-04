from typing import Optional
from pydantic import BaseModel


class Features(BaseModel):
    carat: float
    cut: str
    color: str
    clarity: str
    depth: Optional[float] = None
    table: Optional[float] = None
    x: float
    y: float
    z: float


class Target(BaseModel):
    price: float


class PredictPayload(BaseModel):
    model: str
    features: Features
    target: Optional[Target] = None
