import tensorflow as tf
import numpy as np
import cv2

'''
    This script uses a webcam, and runs the compile .tflite model on it. 
    NOTE: the speed of running the model is completely dependent on your machine and the model you
    trained, and can be very slow
    
    Once it has started, you can quit by pressing 'q'
    
    Author :        Martijn Folmer 
    Date created :  13-01-2023
    
'''


# Path to the compiled tflite with
pathToTflite = 'UNET.tflite'
imgSize_small = (64, 64)
imgSize_big = (256, 256)

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=pathToTflite)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
print("Input details : " + str(input_details))
print("Output details : " + str(output_details))

# Setup the webcam
vid = cv2.VideoCapture(0)
# Set properties. Each returns === True on success (i.e. correct resolution)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:

    # Capture the video frame by frame
    ret, frame = vid.read()   # standard is 480x640

    # turn square in case resolution is 480x640, turn to 480x480
    frame = frame[:, 80:-80]

    # get the small image we run through the tflite
    input_img = cv2.resize(frame, imgSize_small)
    input_img = cv2.resize(input_img, imgSize_big)

    # concatenate, so we can show it
    frame_show = np.concatenate([frame, cv2.resize(input_img, (480, 480))], axis=1)

    # run the interpreter
    interpreter.set_tensor(input_details[0]['index'], [input_img.astype(dtype=np.float32)])
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    # resize output so we can show the results from the model
    output = output[0]
    output = np.clip(output, 0, 255)
    output = cv2.resize(output, (480, 480))
    output = np.asarray(output, dtype=np.uint8)

    # show ground truth, input and output
    frame_show = np.concatenate([frame_show, output], axis=1)

    # Display the resulting frame
    cv2.imshow('frame', frame_show)

    # use q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()


