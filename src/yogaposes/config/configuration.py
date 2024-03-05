from yogaposes.constants import *
from yogaposes.utils.common import read_yaml, create_directories
from yogaposes.entity.config_entity import (DataIngestionConfig, PrepareBaseModelConfig, TrainingConfig)
import os
from dotenv import load_dotenv
load_dotenv()

MLFLOW_TRACKING_URI = os.environ['MLFLOW_TRACKING_URI']
MLFLOW_TRACKING_USERNAME = os.environ['MLFLOW_TRACKING_USERNAME']
MLFLOW_TRACKING_PASSWORD = os.environ['MLFLOW_TRACKING_PASSWORD']


class configurationManager:
    def __init__(self, config_file_path = CONFIG_FILE_PATH, params_file_path = PARAMS_FILE_PATH):
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(root_dir= config.root_dir, source_URL=config.source_URL)

        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        create_directories([config.root_dir])

        get_prepare_base_model_config = PrepareBaseModelConfig(root_dir= config.root_dir,
                                                               resnet_base_model_path=config.resnet_base_model_path,
                                                               resnet_updated_base_model_path= config.resnet_updated_base_model_path,
                                                               params_class=self.params.CLASSES,
                                                               params_image_size= self.params.IMAGE_SIZE,
                                                               params_pretrained = self.params.PRETRAINED
                                                               )

        return get_prepare_base_model_config
    
    def get_traning_config(self) -> TrainingConfig:
        model_training = self.config.model_training
        prepare_base_model = self.config.prepare_base_model
        training_data = os.path.join(self.config.data_ingestion.root_dir, 'yoga-poses-dataset')
        
        create_directories([model_training.root_dir])
        
        training_config = TrainingConfig(root_dir= model_training.resnet_trained_model_path, 
                                        resnet_trained_model_path= model_training.resnet_trained_model_path,
                                        resnet_updated_base_model_path= prepare_base_model.resnet_updated_base_model_path,
                                        traning_data = training_data,
                                        params_augmentation = self.params.AUGMENTATION,
                                        params_image_size = self.params.IMAGE_SIZE,
                                        params_batch_size= self.params.BATCH_SIZE,
                                        params_epoches = self.params.EPOCHS,
                                        params_learning_rate = self.params.LEARNING_RATE,
                                        all_params = self.params,
                                        mlflow_uri= MLFLOW_TRACKING_URI
                                        )
        
        return training_config