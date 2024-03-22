from yogaposes import logger
from yogaposes.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from yogaposes.pipeline.stage_02_prepare_based_model import PrepareBaseModelTrainingPipeline
from yogaposes.pipeline.stage_03_model_training import ModelTrainingPipeline, ModelTrainingPipelineViT



STAGE_NAME = "Data Ingestion"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e 



STAGE_NAME = "Prepare Based Model"

try: 
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_prep = PrepareBaseModelTrainingPipeline()
    model_prep.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\nx==========x")
    
except Exception as e:
    logger.exception(e)
    raise e 

STAGE_NAME = "Model Training"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_trainer = ModelTrainingPipelineViT()
    model_trainer.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e 