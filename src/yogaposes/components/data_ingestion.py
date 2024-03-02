import opendatasets as od
from yogaposes import logger 
from yogaposes.utils.common import get_size
import os
import shutil
from yogaposes.entity.config_entity import (DataIngestionConfig)


class DataIngestion():
    def __init__(self, config:DataIngestionConfig):
        self.config = config 
    
    def download_file(self) -> str:

        try:
            file_name = 'kaggle.json'
            dataset_url = self.config.source_URL
            files_in_folder = os.listdir(self.config.root_dir)

            if file_name not in files_in_folder:
                shutil.copy(os.path.join('research/',file_name), self.config.root_dir)

            logger.info(f'Downloading data from {dataset_url} to {str(self.config.root_dir)}')
            os.chdir(self.config.root_dir)
            od.download(dataset_url)
            os.chdir('../../')
            logger.info(f'Succesfully Downloaded data from {dataset_url} to {str(self.config.root_dir)}')

        except Exception as e:
            raise e