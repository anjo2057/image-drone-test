from run_object_oriented import yolo_model

def main(): 
    # Path to the trained YOLO model and the source video
    model_path = "runs/detect/train4/weights/best_of_all.pt"
    source = "../test_footage/video/flera_dron.mp4"  # Replace with your video path
    # source = "https://www.youtube.com/watch?v=MlFtHuXPbv4"


    # process_video("../test_footage/video/flera_dron.mp4", "runs/detect/train6/weights/best11.pt")


    # Create an instance of the yolo_model class
    yolo = yolo_model(model_path)
    


    yolo_model.run_conf_n_inference(yolo, source=source, stream=True, show=True)

    
if __name__ == "__main__":
    main()