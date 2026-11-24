

# Web Filter 3

Web Filter 3 is a browser-based image editor that uses person segmentation to apply effects selectively. Upload a photo, choose filters, preview the result, and download the processed image. For example, you can blur or replace the background while preserving the person in the foreground.

[Live Demo](https://rahul138188-ops.github.io/WEB-FILTER/) · [API Docs](https://kartikey1234-webfilter-3.hf.space/docs) · [Hugging Face Space](https://huggingface.co/spaces/Kartikey1234/webfilter-3)

## Features

- Blur the background or replace it with a preset or an uploaded image.
- Apply a cartoon-style background from a preset or a custom image.
- Use blur, sharpening, edge detection, brightness adjustment, glare removal, noise reduction, and noise addition.
- Preview the person segmentation mask and adjust intensity for filters that support it.
- Apply multiple filters in one request and download the processed image.

## How it works

1. The HTML, CSS, and JavaScript frontend sends the source image and selected operations to FastAPI as multipart form data.
2. The backend runs Torchvision's pretrained **DeepLabV3 with a ResNet-50 backbone** and extracts the person probability map from the model output.
3. The backend resizes the soft mask to the original image dimensions using bilinear interpolation. For background replacement, it blends the original person region and the new background using the mask as a per-pixel weight.
4. OpenCV and NumPy apply the selected operations in sequence. The backend encodes the result as JPEG and sends it to the browser for preview.

**Stack** · Python, FastAPI, PyTorch, Torchvision, OpenCV, NumPy, Pillow, HTML, CSS, JavaScript, and Docker. The frontend is hosted on GitHub Pages and the API runs on Hugging Face Spaces.

## Run locally

Clone the repository and install the dependencies using Python 3.10.

```bash
git clone https://github.com/rahul138188-ops/WEB-FILTER.git
cd WEB-FILTER
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on macOS/Linux or `.venv\Scripts\Activate.ps1` in Windows PowerShell. Then run:

```bash
python -m pip install -r requirements.txt
python -m uvicorn filterproj.simp:app --host 127.0.0.1 --port 7860
```

The model weights download on first use if they are missing from the Torch cache. Try the API at [http://127.0.0.1:7860/docs](http://127.0.0.1:7860/docs).

To test the frontend with your local API, change the `fetch()` URL in `index.html` to `http://127.0.0.1:7860/filter`. In a second terminal, run `python -m http.server 8000` from the repository root and open [http://localhost:8000](http://localhost:8000).

## API example

`POST /filter` accepts a multipart request with the image in `file` and a JSON array of operations in `values`. Operation names are case-sensitive. For filters with intensity controls, use a value from 0 to 1. The response contains JPEG image bytes.

```bash
curl --fail-with-body http://127.0.0.1:7860/filter \
  -F 'file=@portrait.jpg' \
  -F 'values=[{"name":"Backgroundblur","intensity":0.6}]' \
  --output filtered.jpg
```

You can also pass `background_name` or `cartoon_name` for a preset, or attach a replacement image in `background_file` or `cartoon_file`.

## Repository layout

| Path | Purpose |
| --- | --- |
| `index.html` | Browser interface, filter controls, and API requests |
| `style.css` | Frontend styles |
| `filterproj/simp.py` | FastAPI routes and multipart upload handling |
| `filterproj/seqmodel.py` | Person segmentation, mask blending, and filter operations |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container setup for the backend |

The segmentation pipeline targets the model's **person** class. Background effects work best on photos with a clearly visible person.
