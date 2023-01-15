import cv2
import os

'''
    This is a script in order to create a dataset for the superresolution UNET, using images taken with the webcam
    
    NOTE : as soon as you run this script, the webcam will start taking images. You can quit by stopping the application
    or by pressing q on the keyboard
    NOTE 2 : This is just a simple application to quickly get images from around your house. If you want a more diverse
    dataset, I recommend going to kaggle.com, huggingface.com or the Google Openimages website to get random images
    to mix and match.
    
    Author :        Martijn Folmer 
    Date created :  13-01-2023
'''


# User defined dataset
pathToDataset = ''  # Where we want to save our data
deleteDataset = True                                                # if set to true, we delete all previous training data

# create the folder if it does not yet exist
if not os.path.exists(pathToDataset):
    os.mkdir(pathToDataset)
# delete the dataset if we want to clear the folder of previously made training data.
if deleteDataset:
    [os.remove(pathToDataset + "/" + f) for f in os.listdir(pathToDataset)]

# how many images are currently in the dataset
kn = len(os.listdir(pathToDataset))

# Setup the webcam
vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:

    # Capture the video frame by frame
    ret, frame = vid.read()   # standard is 480x640

    # turn square in case resolution is 480x640, turn to 480x480
    frame = frame[:, 80:-80]

    # Saving the images in the folder wherew we want our dataset
    cv2.imwrite(pathToDataset + f"/img_{kn}.png", frame)
    kn += 1

    # Display the resulting frame
    image = cv2.putText(frame, f"Tot Num Img : {kn}", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow('frame', frame)

    # use q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
