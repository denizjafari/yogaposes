from yogaposes.pipeline.prediction import PosePrediction

if __name__ == "__main__":
    try:
        prediction = PosePrediction()
        image_path = input("Enter Image File Path: ")
        c= prediction.predict(image_path)
        prediction.display_image(image_path)
        print('The predicted pose is: ', c)
    except Exception as e:
        raise e
        