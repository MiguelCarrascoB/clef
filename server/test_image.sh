#!/bin/bash
# Image-capability test: downloads a sample image, base64's it, sends it through /v1/systemone.
source ~/venvs/clef/bin/activate
export HSA_ENABLE_DXG_DETECTION=1
cd /tmp
wget -q -O test.jpg https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG
python - << 'EOF'
import base64, json, urllib.request

with open("/tmp/test.jpg", "rb") as fh:
    b64 = base64.b64encode(fh.read()).decode()

request = {
    "model": "clef-flash",
    "state": "A photo is shown below.",
    "images": ["data:image/jpeg;base64," + b64],
    "questions": {
        "has_candy": {"type": "noul", "instructions": "Does the photo show candy or sweets?"},
        "animal": {"type": "choice", "instructions": "What animal is depicted on the candy packaging?",
                   "criteria": {"bird": "A bird", "bear": "A bear", "none": "No animal"}},
        "colorful": {"type": "score", "instructions": "How colorful is the photo?",
                     "criteria": ["Muted", "Moderate", "Vivid"]},
    },
}
req = urllib.request.Request(
    "http://localhost:8910/v1/systemone",
    data=json.dumps(request).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=120) as res:
    out = json.loads(res.read())
print(json.dumps(out, indent=2))
assert out["answers"]["has_candy"]["noul"] > 0.5, "should detect candy"
EOF
echo "IMAGE TEST DONE"
