import sys


cap = cv2.VideoCapture('Videli Free Footage - Different Casual People Walking 720.mp4')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'H264')
out = cv2.VideoWriter('runs/output.mp4', fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    results = model.track(frame, persist=True)
    annotated_frame = results[0].plot()
    out.write(annotated_frame)

cap.release()
out.release()
cv2.destroyAllWindows()

print("Video saved as output.avi")


