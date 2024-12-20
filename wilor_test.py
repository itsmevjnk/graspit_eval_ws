import torch
import cv2
import sys
import open3d as o3d

from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float32

pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)

# Open a connection to the camera (0 is the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    sys.exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the resulting frame
    cv2.imshow('Camera Feed', frame)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    outputs = pipe.predict(frame_rgb)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
