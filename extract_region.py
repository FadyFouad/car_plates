import torch
from torchvision import transforms
from PIL import Image

from mobilenet_unet import MobileNetUNet

import matplotlib.pyplot as plt
import torch.nn.functional as fun

image_path = "data/test/001.jpg"

# Load the model
model = MobileNetUNet(num_classes=1)
model.load_state_dict(torch.load("models/mobilenet_unet.pth", map_location=torch.device('cpu')))
model.eval()

# Define preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Adjust to your model's input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
])

# Load a sample image
image = Image.open(image_path).convert("RGB")
input_image = transform(image).unsqueeze(0)  # Add batch dimension

print("Input image size:", input_image.shape)  # Should be [1, 3, H, W]

# Inference
with torch.no_grad():
    output = model(input_image)

# Resize the output to match the input image dimensions
predicted_mask = fun.interpolate(output, size=(256, 256), mode="bilinear", align_corners=False)
print("Output size:", predicted_mask.shape)  # Should be [1, 1, 256, 256]

# Apply sigmoid and threshold
predicted_mask = torch.sigmoid(predicted_mask).squeeze().cpu().numpy()
predicted_mask = (predicted_mask > 0.5).astype("uint8")

# Debugging shapes
print("Upsampled output size:", predicted_mask.shape)  # Should be [256, 256]


# Display the input image and the predicted mask
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Input Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Predicted Mask")
plt.imshow(predicted_mask, cmap="gray")
plt.axis("off")

plt.show()

#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the original image
image = cv2.imread(image_path)

# Scale mask if values are in [0, 1]
if predicted_mask.max() <= 1.0:
    predicted_mask = (predicted_mask * 255).astype(np.uint8)

# Resize mask to match image dimensions
predicted_mask = cv2.resize(predicted_mask, (image.shape[1], image.shape[0]))

# Ensure the mask is binary
predicted_mask = (predicted_mask > 127).astype(np.uint8) * 255

# Apply the mask to the original image
masked_image = cv2.bitwise_and(image, image, mask=predicted_mask)

# crop masked image
x, y, w, h = cv2.boundingRect(predicted_mask)
cropped_masked_image = masked_image[y:y+h, x:x+w]


# Visualize the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Masked Image")
plt.imshow(cv2.cvtColor(cropped_masked_image, cv2.COLOR_BGR2RGB))
plt.axis("off")


plt.tight_layout()
plt.show()

# save the masked image
cv2.imwrite("data/output/masked_image.jpg", cropped_masked_image)
