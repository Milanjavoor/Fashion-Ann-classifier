from __future__ import annotations

import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import logging
from typing import Sequence


NUM_CLASSES=10
IMAGE_SIZE=28
IMAGE_SHAPE=(IMAGE_SIZE,IMAGE_SIZE,1)
CLASS_NAMES=("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")

split=tuple[np.ndarray,np.ndarray]
logger=logging.getLogger(__name__)

def validate(x:np.ndarray,y: np.ndarray,name:str):
    if x.ndim != 3 or x.shape[1:]!=(IMAGE_SIZE,IMAGE_SIZE):
        raise ValueError(f"{name}:expected (N,28,28) images, got {x.shape}")
    if x.dtype != np.uint8:
        raise ValueError("Got unexpected data type")
    if y.shape!=(x.shape[0],):
        raise ValueError ("Label shapes do not match ")
    if y.min() <0 or y.max()>=NUM_CLASSES:
        raise ValueError(f"labels must be in [0,{NUM_CLASSES}]")

    
def split_train_val(x:np.ndarray,y:np.ndarray,val_fraction,seed:int):
    x_tr,x_va,y_tr,y_va=train_test_split(x,y,test_size=val_fraction,stratify=y,random_state=seed)
    return (x_tr,y_tr),(x_va,y_va)

def load_fashion_mnist(val_fraction:float,seed:int)->tuple[split,split,split]:
    (x,y),(x_test,y_test)=tf.keras.datasets.fashion_mnist.load_data()
    validate(x,y,"train")
    validate(x_test,y_test,"test")
    train,val=split_train_val(x,y,val_fraction,seed)
    logger.info("Splits: train=%d val=%d test=%d",len(x_test),len(val[0]),len(train[0]))
    return train,val,(x_test,y_test)
def to_model_input(x:np.ndarray)->np.ndarray:
    return (np.asarray(x,dtype=np.float32)/255.0)[...,np.newaxis]

def _augment(image,label)->tuple[tf.Tensor,tf.Tensor]:
    image=tf.image.random_flip_left_right(image)
    image=tf.pad(image,[[2,2],[2,2],[0,0]],mode="CONSTANT")
    image=tf.image.random_crop(image,size=IMAGE_SHAPE)
    return image,label

def make_dataset(split:split,batch_size:int,training:bool,augment:bool=False,seed:int=42)->tf.data.Dataset:
    x,y=split
    ds=tf.data.Dataset.from_tensor_slices((to_model_input(x),y.astype(np.int32)))
    ds=ds.map(lambda img,lbl: (img,tf.one_hot(lbl,NUM_CLASSES)),
              num_parallel_calls=tf.data.AUTOTUNE,)
    ds=ds.cache()
    if training:
        ds=ds.shuffle(buffer_size=len(x),seed=seed,reshuffle_each_iteration=True)
        if augment:
            ds=ds.map(_augment,num_parallel_calls=tf.data.AUTOTUNE)
    ds=ds.batch(batch_size,drop_remainder=training)
    return ds.prefetch(tf.data.AUTOTUNE)






