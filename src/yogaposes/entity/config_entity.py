# working on the entity

from dataclasses import dataclass # ensure that the output is what you set to be
from pathlib import Path

# constructor class, get input and make it like this variable
# value of the key in the yml file needs to be of the enforced type below
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    
    
@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    resnet_base_model_path: Path
    resnet_updated_base_model_path: Path
    params_class: int
    params_image_size: list
    params_pretrained: bool


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path 
    resnet_trained_model_path: Path
    resnet_updated_base_model_path: Path
    traning_data: Path
    params_augmentation: bool
    params_image_size: list 
    params_batch_size: int 
    params_epoches: int
    params_learning_rate: float
    all_params: dict
    mlflow_uri: str