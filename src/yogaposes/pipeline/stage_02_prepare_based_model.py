from yogaposes.config.configuration import configurationManager
from yogaposes.components.prepare_base_model import PrepareBaseModel
from yogaposes import logger

STAGE_NAME = "Prepare Based Model"



class PrepareBaseModelTrainingPipeline:

    def __init__(self) -> None:
        pass

    def main(self): 
        config = configurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.update_base_model()




if __name__ == "__main__":
    try:
    
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e 