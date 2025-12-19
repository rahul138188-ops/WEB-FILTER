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
import numpy as np
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = DeepLabV3_ResNet50_Weights.DEFAULT
model = deeplabv3_resnet50(weights=weights)
model.eval()
model.to(device)

def Blurring(inputmask,inputimage):
  blurred_img=cv2.GaussianBlur(inputimage,(15,15),0)
  filtered_img=inputmask*blurred_img+(1-inputmask)*inputimage
  return filtered_img

def Sharpen(inputmask,inputimage):
  kernel=np.array([[0,-1,0],
                [-1,5,-1],
                [0,-1,0]],dtype=np.float32)
  smooth_img=cv2.filter2D(inputimage,-1,kernel)
  filterd_img=inputmask*smooth_img+(1-inputmask)*inputimage
  return filterd_img

def EdgeDetect(inputmask,inputimage):
    low=100
    high=200
    gray = cv2.cvtColor(inputimage, cv2.COLOR_RGB2GRAY)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    edges = cv2.Canny(gray, low, high)
    # Convert to 3-channel
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return edges_color.astype(np.float32)

def Backgroundblur(inputmask,inputimage):
  blurred_img=cv2.GaussianBlur(inputimage,(15,15),0)
  filtered_img=inputmask*inputimage+(1-inputmask)*blurred_img
  return filtered_img

def changebackground(inputmask,inputimage):
  script_dir = os.path.dirname(os.path.abspath(__file__))
  image_path = os.path.join(script_dir, "background.jpeg")
  background_image=Image.open(image_path).convert("RGB")
  img = np.array(background_image.resize((inputimage.shape[1], inputimage.shape[0]))).astype(np.float32)
  filttered_img=inputmask*inputimage+(1-inputmask)*img
  return filttered_img

def Brightness(inputmask,inputimage):
    kernel=np.array([[0,0,0],
                [0,1,0],
                [0,0,0]],dtype=np.float32)

    bright_image = cv2.filter2D(inputimage, -1, kernel)
    bright_image = cv2.add(bright_image, 50)
    return bright_image

def Segmentaion(inputmask, inputimage):
    """
    Returns the object mask as an RGB image.
    inputmask: (H,W,1) float32 between 0-1
    inputimage: original image (unused)
    """
    # Convert mask to 0-255
    mask_uint8 = (inputmask[..., 0] * 255).astype(np.uint8)  # (H,W)
    # Make it 3-channel for RGB display
    mask_rgb = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2RGB)
    return mask_rgb.astype(np.float32)

def Cartoon(inputmask,inputimage):
  script_dir = os.path.dirname(os.path.abspath(__file__))
  image_path = os.path.join(script_dir, "cartoon.jpg")
  background_image=Image.open(image_path).convert("RGB")
  img = np.array(background_image.resize((inputimage.shape[1], inputimage.shape[0]))).astype(np.float32)
  filttered_img=inputmask*inputimage+(1-inputmask)*img
  return filttered_img

def glareremoval(inputmask, inputimage):
    # Ensure input image is uint8
    img = inputimage.astype(np.uint8) if inputimage.dtype != np.uint8 else inputimage.copy()
    
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # if your image is RGB
    v = hsv[:, :, 2]
    
    # Threshold glare
    _, glare_mask = cv2.threshold(v, 220, 255, cv2.THRESH_BINARY)
    
    # Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_CLOSE, kernel)
    
    # Ensure mask is 8-bit single channel
    glare_mask = glare_mask.astype(np.uint8)
    
    # Inpaint
    result = cv2.inpaint(img, glare_mask, 5, cv2.INPAINT_TELEA)
    
    # Convert back to float32 if needed
    return result.astype(np.float32)


def Noisereduciton(inputmask,inputimage):
  mask = cv2.GaussianBlur(inputmask, (5,5), 0).astype(np.float32)
  mask[mask < 0.1] = 0
  if mask.ndim == 2:
   mask=mask[...,None]
  inputimage=inputimage.astype(np.float32)
  blurred_img = cv2.GaussianBlur(inputimage, (5,5), 0)
  filtered_img = mask * blurred_img + (1 - mask) * inputimage
  return filtered_img


def Noiseaddition(inputmask, inputimage):
    """
    Adds Gaussian noise to the masked region of the image.
    
    Parameters:
        inputmask: np.array of shape (H,W,1) or (H,W) with values 0-1
        inputimage: np.array of shape (H,W,3) in float32
        mean: mean of Gaussian noise
        std: standard deviation of Gaussian noise
    
    Returns:
        np.array of same shape as inputimage, dtype float32
    """
    mean=0
    std=10
    # Ensure mask has shape (H,W,1)
    mask = inputmask
    if mask.ndim == 2:
        mask = mask[..., None]

    # Make sure image is float32
    img = inputimage.astype(np.float32)

    # Generate Gaussian noise
    noise = np.random.normal(mean, std, img.shape).astype(np.float32)

    # Add noise only to the masked region
    noisy_img = mask * (img + noise) + (1 - mask) * img

    return noisy_img

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

def processimage(image_bytes: bytes,operations):
  img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
  transform=weights.transforms()
  input_tensor=transform(img).unsqueeze(0).to(device)
  with torch.no_grad():
    output = model(input_tensor)["out"]
    pred_mask = torch.softmax(output, dim=1).squeeze(0)
    print(pred_mask.shape)

  probmask=pred_mask[15,:,:]
  img_or=np.array(img)
  H,W=img_or.shape[:2]
  print(img_or.shape)
  mask_prob = F.interpolate(
    probmask.unsqueeze(0).unsqueeze(0),  # (1,1,h,w)
    size=(H, W),
    mode="bilinear",
    align_corners=False
   )[0, 0]  # back to (H, W)
  print(type(mask_prob))
  mask_prob = mask_prob.detach().cpu().numpy()
  print(type(mask_prob))
  img_or=np.array(img).astype(np.float32)
  mask_prob=mask_prob[...,None]
  print(mask_prob.shape)
  op_results = []

  for op_name in operations:
    func = OPERATION_MAP[op_name]
    op_result = func(mask_prob, img_or).astype(np.float32)
    op_results.append(op_result)

# Combine equally
  num_ops = len(op_results)
  final_img = np.zeros_like(img_or, dtype=np.float32)

  for op_result in op_results:
    final_img += op_result / num_ops  # no extra mask multiplication

  final_img_uint8 = np.clip(final_img, 0, 255).astype(np.uint8)
  pilimg = Image.fromarray(final_img_uint8)
  buf = io.BytesIO()
  pilimg.save(buf, format="JPEG")
  buf.seek(0)
  return buf.read()





