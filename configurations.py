from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Sequence


@dataclasses.dataclass(frozen=True)
class Config:
    seed:int=42
    val_frac:float=0.1
    batch_size:int=128
    epochs:int=100
    hidden_units:tuple[int,...]=(512,256,128)
    droput:float=0.3
    l2:float=1e-4
    label_smoothing:float=0.1
    peak_lr:float=2e-3
    min_lr:float=1e-6
    warmup_epochs=3
    weight_decay=1e-4
    patience=15
    argument=True
    mixed_precision=False
    output_dir:Path=Path("runs/baseline")


    def __post_init__(self) -> None:
        if not 0.0<self.val_frac<0.5:
            raise ValueError(f"val fraction must be in  (0,0.5)")
        if self.batch_size<=0 or self.epochs<=0:
            raise ValueError ("Batch size and epochs must be positive")
        if not 0.0<self.droput<1.0:
            raise ValueError("Droput value is not between [0,1]")
        if not 0.0<self.label_smoothing<1.0:
            raise ValueError("Label smoothing value should be in  betweeen (0,1)")
        if not 0.0<self.min_lr<self.peak_lr:
            raise ValueError("Minimum learning rate is invalid")   
        if not 0<self.warmup_epochs<self.epochs:
            raise ValueError("Number of warmup epochs is invalid") 
        if not self.hidden_units or any(u<=0 for u  in self.hidden_units):
            raise ValueError("Hidden units must be non-empty and positive")

    def to_json(self) -> str:
        return json.dumps(dataclasses(self),indent=2,default=str)


def int_tuple(text:str) -> tuple[int,...]:
    return tuple(int(v) for v in text.split(",") if v.strip())


def parse_args(argv:Sequence[str] | None=None) ->Config:
    d=Config()
    p=argparse.ArgumentParser(description="Train a fully conneected classifier on Fashion-MNIST")
    p.add_argument("--seed",type=int,default=d.seed)
    p.add_argument("--val-fraction",type=float,default=d.val_frac)
    p.add_argument("--batch-size",type=int,default=d.batch_size)
    p.add_argument("--epochs",type=int,default=d.epochs)
    p.add_argument("--hidden-units",type=_int_tuple,default=d.hidden_units,help=
                   "comma-separated layer widths ,e.g 512,256,128")
    p.add_argument("--dropout",type=float,default=d.droput)
    p.add_argument("--l2",type=float,default=d.l2)
    p.add_argument("--label-smoothing",type=float,default=d.label_smoothing)
    p.add_argument("--peak-lr",type=float,default=d.peak_lr)
    p.add_argument("--min-lr",type=float,default=d.min_lr)
    p.add_argument("--warmup-epochs",type=float,default=d.warmup_epochs)
    p.add_argument("--weight-decay",type=float,default=d.weight_decay)
    p.add_argument("--patience",type=int,default=d.patience)
    p.add_argument("--no-augment",action="store_true")
    p.add_argument("--mixed-precision",action="store_true")
    p.add_argument("--output-dir",type=Path,default=d.output_dir)
    a=p.parse_args(argv)
    return Config(
        seed=a.seed , val_frac=a.val_frac, batch_size=a.batch_size,
        epochs=a.epochs , hidden_units=a.hidden_units, droput=a.dropout,
        l2=a.l2, label_smoothing=a.label_smoothing , peak_lr=a.peak_lr ,
        min_lr=a.min_lr, weight_decay=a.weight_decay, warmup_epochs=a.warmup_epochs,
        patience=a.patience,augment= not a.no_augment ,mixed_precision=a.mixed_precision,
        output_dir=a.output_dir
    )



