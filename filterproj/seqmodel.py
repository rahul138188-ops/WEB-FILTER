import torch
import torchvision.transforms as T
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights
)
import cv2
import os
import io
import torch.nn.functional as F
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = DeepLabV3_ResNet50_Weights.DEFAULT
model = deeplabv3_resnet50(weights=weights)
model.eval()
model.to(device)
# ----------------- FILTER FUNCTIONS -------------------
def Blurring(inputmask, inputimage, intensity=0.5):
    """
    intensity: 0-1, fraction of max blur applied
    """
    ksize = int(5 + intensity * 30)  # kernel size from 5 to 35
    if ksize % 2 == 0:
        ksize += 1
    blurred_img = cv2.GaussianBlur(inputimage, (ksize, ksize), 0)
    filtered_img = inputmask * blurred_img + (1 - inputmask) * inputimage
    return filtered_img.astype(np.float32)


def Sharpen(inputmask, inputimage, intensity=0.5):
    """
    intensity: 0-1, fraction of sharpening applied
    """
    kernel = np.array([[0, -intensity, 0],
                       [-intensity, 1 + 4 * intensity, -intensity],
                       [0, -intensity, 0]], dtype=np.float32)
    smooth_img = cv2.filter2D(inputimage, -1, kernel)
    filtered_img = inputmask * smooth_img + (1 - inputmask) * inputimage
    return filtered_img.astype(np.float32)


def EdgeDetect(inputmask, inputimage, intensity=None):
    """
    intensity not used
    """
    low = 100
    high = 200
    gray = cv2.cvtColor(inputimage, cv2.COLOR_RGB2GRAY)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    edges = cv2.Canny(gray, low, high)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return edges_color.astype(np.float32)


def Backgroundblur(inputmask, inputimage, intensity=0.5,):
    ksize = int(5 + intensity * 30)
    if ksize % 2 == 0:
        ksize += 1
    blurred_img = cv2.GaussianBlur(inputimage, (ksize, ksize), 0)
    filtered_img = inputmask * inputimage + (1 - inputmask) * blurred_img
    return filtered_img.astype(np.float32)

def changebackground(
    inputmask,
    inputimage,
    background="background.jpg",
    background_img=None  # NEW
):
    # ---------- Priority: uploaded image ----------
    if background_img is not None:
        img = cv2.resize(
            background_img,
            (inputimage.shape[1], inputimage.shape[0])
        ).astype(np.float32)

    else:
        # ---------- Fallback: file name ----------
        if background == "" or background is None:
            background = "background.jpg"

        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_dir, background)

        if not os.path.exists(image_path):
            print("Background image not found:", image_path)
            return inputimage.astype(np.float32)

        background_image = Image.open(image_path).convert("RGB")
        img = np.array(
            background_image.resize(
                (inputimage.shape[1], inputimage.shape[0])
            )
        ).astype(np.float32)

    # ---------- Blend ----------
    filtered_img = inputmask * inputimage + (1 - inputmask) * img
    return filtered_img.astype(np.float32)



def Brightness(inputmask, inputimage, intensity=0.5):
    """
    intensity: 0-1, fraction of max brightness add
    """
    add_value = int(intensity * 100)  # max add 100
    bright_image = cv2.add(inputimage, add_value)
    return bright_image.astype(np.float32)


def Segmentaion(inputmask, inputimage, intensity=None):
    mask_uint8 = (inputmask[..., 0] * 255).astype(np.uint8)
    mask_rgb = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2RGB)
    return mask_rgb.astype(np.float32)


def Cartoon(
    inputmask,
    inputimage,
    cartoon="cartoon.jpg",
    cartoon_img=None   # NEW
):
    # ---------- Priority: uploaded image ----------
    if cartoon_img is not None:
        img = cv2.resize(
            cartoon_img,
            (inputimage.shape[1], inputimage.shape[0])
        ).astype(np.float32)

    else:
        # ---------- Fallback: file name ----------
        if cartoon == "" or cartoon is None:
            cartoon = "cartoon.jpg"

        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_dir, cartoon)

        if not os.path.exists(image_path):
            print("Cartoon image not found:", image_path)
            return inputimage.astype(np.float32)

        background_image = Image.open(image_path).convert("RGB")
        img = np.array(
            background_image.resize(
                (inputimage.shape[1], inputimage.shape[0])
            )
        ).astype(np.float32)

    # ---------- Blend ----------
    filtered_img = inputmask * inputimage + (1 - inputmask) * img
    return filtered_img.astype(np.float32)

def glareremoval(inputmask, inputimage, intensity=None):
    img = np.clip(inputimage, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    bright = v > 215
    low_sat = s < 45
    glare_mask = (bright & low_sat).astype(np.uint8) * 255
    glare_mask = cv2.GaussianBlur(glare_mask, (15, 15), 0)
    smooth = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    mask = glare_mask.astype(np.float32) / 255.0
    mask = mask[..., None]
    result = mask * smooth + (1 - mask) * img
    return result.astype(np.float32)


def Noisereduciton(inputmask, inputimage, intensity=0.5):
    """
    intensity: 0-1, fraction of max blur in noise reduction
    """
    mask = cv2.GaussianBlur(inputmask, (5, 5), 0).astype(np.float32)
    mask[mask < 0.1] = 0
    if mask.ndim == 2:
        mask = mask[..., None]
    blurred_img = cv2.GaussianBlur(inputimage, (5, 5), 0)
    filtered_img = mask * (blurred_img * intensity + inputimage * (1 - intensity)) + (1 - mask) * inputimage
    return filtered_img.astype(np.float32)


def Noiseaddition(inputmask, inputimage, intensity=0.5):
    """
    intensity: 0-1, fraction of noise applied
    """
    mean = 0
    std = intensity * 50  # max std = 50
    mask = inputmask
    if mask.ndim == 2:
        mask = mask[..., None]
    img = inputimage.astype(np.float32)
    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy_img = mask * (img + noise) + (1 - mask) * img
    return noisy_img.astype(np.float32)


# ----------------- OPERATION MAP -------------------
OPERATION_MAP = {
    "Blurring": Blurring,
    "Sharpen": Sharpen,
    "EdgeDetect": EdgeDetect,
    "Backgroundblur": Backgroundblur,
    "changeBackground": changebackground,
    "Brightness": Brightness,
    "Segmentation": Segmentaion,
    "Cartoon": Cartoon,
    "glareremoval": glareremoval,
    "Noisereduction": Noisereduciton,
    "Noiseaddition": Noiseaddition,
}



# ----------------- PROCESS IMAGE -------------------
def processimage(
    image_bytes: bytes,
    operations,
    cartoon_name="",
    background_name="",
    cartoon_bytes=None,
    background_bytes=None
):
    # -------- Load main image --------
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = weights.transforms()
    input_tensor = transform(img).unsqueeze(0).to(device)

    # -------- Segmentation --------
    with torch.no_grad():
        output = model(input_tensor)["out"]
        pred_mask = torch.softmax(output, dim=1).squeeze(0)

    probmask = pred_mask[15, :, :]  # person class
    img_or = np.array(img).astype(np.float32)
    H, W = img_or.shape[:2]

    mask_prob = F.interpolate(
        probmask.unsqueeze(0).unsqueeze(0),
        size=(H, W),
        mode="bilinear",
        align_corners=False
    )[0, 0]

    mask_prob = mask_prob.detach().cpu().numpy()[..., None]

    final_img = img_or.copy()

    # -------- Load optional cartoon / background --------
    cartoon_img = None
    background_img = None

    if cartoon_bytes:
      cartoon_img = cv2.imdecode(
        np.frombuffer(cartoon_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )
    if cartoon_img is not None:

        cartoon_img = cv2.cvtColor(cartoon_img, cv2.COLOR_BGR2RGB)
        cartoon_img = cartoon_img.astype(np.float32)

    if background_bytes:
     background_img = cv2.imdecode(
        np.frombuffer(background_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )
    if background_img is not None:
        background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
        background_img = background_img.astype(np.float32)

    # -------- Apply operations --------
    for op in operations:
        name = op.get("name")
        intensity = op.get("intensity", 0.5)
        func = OPERATION_MAP[name]

        if name == "changeBackground":
         final_img = func(
         mask_prob,
         final_img,
         background=background_name,   # ✅ correct name
         background_img=background_img
          )


        elif name == "Cartoon":
         final_img = func(
         mask_prob,
         final_img,
         cartoon=cartoon_name,   # ✅ correct name
         cartoon_img=cartoon_img
           )


        else:
            final_img = func(
                mask_prob,
                final_img,
                intensity=intensity
            )

    # -------- Output --------
    final_img_uint8 = np.clip(final_img, 0, 255).astype(np.uint8)
    pilimg = Image.fromarray(final_img_uint8)

    buf = io.BytesIO()
    pilimg.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()
