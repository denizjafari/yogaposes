from pathlib import Path
import torch
from PIL import ImageFile, Image
from torchvision.transforms import Compose, ToTensor, Normalize, \
Resize, CenterCrop
import numpy as np
import os
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import TrainingArguments, Trainer




class PosePrediction(): 
    def __init__(self, base_model_path=None, train_model_path=None):
        
        self.base_model_path = base_model_path if base_model_path else Path('artifacts/prepare_base_model/resnet_updated_base_model.pth')
        self.trained_model_path = train_model_path if train_model_path else Path('artifacts/training/resnet_model.pth')
        self.model = self.load_model()
        self.load_checkpoint()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #self.device = 'cpu'
        self.model.to(self.device)
        
        
    def load_model(self): 
        return torch.load(self.base_model_path)
    
    def load_checkpoint(self):
        checkpoint = torch.load(self.trained_model_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])   
        self.model.train() # always use train for resuming traning
        
    def _preprocess_image(self, filename):
        image = Image.open(filename)
        # Allow loading of truncated images
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        # image net statistics
        normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
        
        composer = Compose([Resize(256), CenterCrop(224), ToTensor(), normalizer])
        image = composer(image).unsqueeze(0)
        return image
        
    def display_image(self,image_path):
        try:
        # Open the image file
            img = Image.open(image_path)
        # Display the image
            img.show()
        except IOError:
            print(f"Error opening the image file at {image_path}. Please ensure the file exists and is an image.")

        
    def predict(self,x_filename):
        
        self.load_checkpoint()
        self.model.eval()
        x = self._preprocess_image(x_filename)
        x_tensor = torch.as_tensor(x).float()
        y_hat_tensor = self.model(x_tensor.to(self.device))
        
        # set it back to the train mode
        self.model.train()
        
        labels = {0:'downdog', 1: 'godess', 2:'plank', 3:'tree', 4:'warrior2'}
        prediction=np.argmax(y_hat_tensor.detach().cpu().numpy())
        
        return labels[prediction]
    

class PredictionPipelineViT:
    def __init__(self, trained_model_path=None):
        self.trained_model_path = trained_model_path if trained_model_path else Path('artifacts/training/')
        self.processor, self.model = self.load_checkpoint()
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
        self.model.to(self.device)
    
    
    def load_model(self, dataset):
        model_name = "google/vit-base-patch16-224"
        # mapping integer labels to string labels and vv
        id2label = dict((k,v) for k,v in enumerate(dataset['train'].features['label'].names))
        label2id = dict((v,k) for k,v in enumerate(dataset['train'].features['label'].names))
        processor = ViTImageProcessor.from_pretrained(model_name)
        model = ViTForImageClassification.from_pretrained(model_name, num_labels=len(id2label), ignore_mismatched_sizes=True, id2label=id2label, label2id=label2id)
        return processor, model
    
    def load_checkpoint(self):
        processor = ViTImageProcessor.from_pretrained(self.trained_model_path)
        model = ViTForImageClassification.from_pretrained(self.trained_model_path)
        return processor, model
    
    
    def _preprocess_image(self, filename):
        # Load the image and apply the transformation
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image = Image.open(filename)
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs
    
    
    def predict(self, filename):
        self.load_checkpoint()
        preprocess_image = self._preprocess_image(filename)
        preprocess_image.to(self.device)
        outputs = self.model(**preprocess_image)
        logits = outputs.logits
        # model predicts one of the 1000 ImageNet classes
        predicted_class_idx = logits.argmax(-1).item()
        
        return self.model.config.id2label[predicted_class_idx]
    
'''    
if __name__ == "__main__":
    try:
        prediction = PosePrediction()
        image_path = input("Enter Image File Path:")
        c= prediction.predict(image_path)
        print('The predicted pose is: ', c)
    except Exception as e:
        raise e
        
        
'''