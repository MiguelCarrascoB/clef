#!/bin/bash
# Video-capability test: generate a tiny mp4 (synthetic frames), send through /v1/systemone.
source ~/venvs/clef/bin/activate
export HSA_ENABLE_DXG_DETECTION=1
python - << 'EOF'
import base64, io, json, urllib.request
import imageio.v3 as iio
import numpy as np

# 8 frames, 320x320: a red ball moving across a dark background
frames = []
for i in range(8):
    img = np.zeros((320, 320, 3), dtype=np.uint8)
    img[:, :, 0] = 20
    x = 20 + i * 35
    img[130:190, x:x + 60, 0] = 255
    frames.append(img)
buf = io.BytesIO()
iio.imwrite(buf, np.stack(frames), extension=".mp4", fps=2)
b64 = base64.b64encode(buf.getvalue()).decode()
print("video bytes:", len(buf.getvalue()))

request = {
    "model": "clef-flash",
    "state": "A short video clip is attached.",
    "videos": ["data:video/mp4;base64," + b64],
    "questions": {
        "has_ball": {"type": "noul", "instructions": "Does a moving red object appear in the video?"},
        "ball_color": {"type": "choice", "instructions": "Which color is the moving object?",
                       "criteria": {"red": "Red", "green": "Green", "blue": "Blue"}},
    },
}
req = urllib.request.Request(
    "http://localhost:8910/v1/systemone",
    data=json.dumps(request).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=600) as res:
    out = json.loads(res.read())
print(json.dumps(out["answers"], indent=2))
assert out["answers"]["has_ball"]["noul"] > 0.5, "should detect the moving object"
assert out["answers"]["ball_color"]["choice"] == "red", "object should be seen as red"
print("VIDEO TEST PASSED")
EOF
