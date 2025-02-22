import cv2
import requests
import time

STREAM_URL = "https://lfh20130915--livestream-basic-stream-me.modal.run"

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't read frame.")
            break

        _, frame_bytes = cv2.imencode('.jpg', frame)

        try:
            print(frame_bytes.tobytes())
            print("\n")
            print("\n")
            print("\n")
            response = requests.post(
                STREAM_URL,
                data=frame_bytes.tobytes(),
                headers={'Content-Type': 'image/jpeg'},
                stream=True
            )
            response.raise_for_status()

            if response.headers['Content-Type'] == 'multipart/x-mixed-replace; boundary=frame':
                for chunk in response.iter_content(chunk_size=None):
                    if chunk.startswith(b'--frame'): # MJPEG boundary
                        image_data = chunk.split(b'\r\n\r\n', 1)[1].rstrip(b'\r\n')
                        if image_data:
                            nparr = np.frombuffer(image_data, np.uint8)
                            processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if processed_frame is not None:
                                cv2.imshow('Live Stream', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit
                break

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            time.sleep(1) # Wait and retry after error

finally:
    cap.release()
    cv2.destroyAllWindows()