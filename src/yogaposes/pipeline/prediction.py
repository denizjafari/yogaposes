from pathlib import Path
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, CenterCrop
import numpy as np
from PIL import ImageFile, Image



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