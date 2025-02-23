import modal

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "python3-opencv", "ffmpeg")
    .pip_install(
        "opencv-python==4.10.0.84",
        "ffmpeg-python==0.2.0",
        "fastapi[standard]",
        "numpy"
    )
)
app = modal.App("livestream-basic", image=image)

@app.function()
def process_frame(frame_bytes):
    return frame_bytes

@app.function()
@modal.web_endpoint(method="POST")
def stream_me(request):
    import cv2
    from fastapi.responses import StreamingResponse
    import numpy as np

    async def frame_generator():
        async for chunk in request.stream():
            nparr = np.frombuffer(chunk, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                print("Error decoding frame")
                continue

            processed_frame_bytes = process_frame.remote(cv2.imencode('.jpg', frame)[1].tobytes())

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + processed_frame_bytes + b'\r\n')

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

