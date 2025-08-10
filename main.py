import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Function to Compute Asymmetry Feature & Extract Skull Region
# ----------------------------------------------------------
def compute_asymmetry_feature(image):
    """
    Given a grayscale CT image (numpy array), compute an asymmetry feature
    and extract the skull region.

    Returns:
        mean_diff: Mean absolute difference between left and mirrored right hemispheres.
        skull_region: Cropped image (numpy array) corresponding to the skull.
    """
    height, width = image.shape

    # Preprocess: Gaussian blur and thresholding
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]

    # Create kernel using NumPy
    kernel_np = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_np)

    # Find contours and select the largest (assumed skull contour)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if not contours:
        return 0.0, image  # fallback: use whole image if no contour found
    skull_contour = contours[0]

    # Fit ellipse to the skull contour
    ellipse = cv2.fitEllipse(skull_contour)
    (center_x, center_y), (major_axis, minor_axis), angle = ellipse

    # Create an elliptical mask
    ellipse_mask = np.zeros_like(image, dtype=np.uint8)
    cv2.ellipse(ellipse_mask, ellipse, color=255, thickness=-1)

    # Create coordinate grids using NumPy
    y_coords, x_coords = np.indices((height, width))
    angle_rad = np.deg2rad(angle)
    perp_angle_rad = angle_rad + np.pi/2  # perpendicular direction

    dx = x_coords - center_x
    dy = y_coords - center_y
    D = -np.sin(perp_angle_rad) * dx + np.cos(perp_angle_rad) * dy

    # Define left/right masks, restricted to the skull region
    left_mask = (D >= 0) & (ellipse_mask.astype(bool))
    right_mask = (D < 0) & (ellipse_mask.astype(bool))

    # Process hemispheres using the original image
    left_hemisphere_full = np.zeros_like(image)
    right_hemisphere_full = np.zeros_like(image)
    left_hemisphere_full[left_mask] = image[left_mask]
    right_hemisphere_full[right_mask] = image[right_mask]

    # Compute bounding box of the skull region
    ys, xs = np.where(ellipse_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0.0, image  # fallback if no mask found
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    skull_region = image[y_min:y_max+1, x_min:x_max+1]

    # Crop hemispheres to the bounding box
    left_hemisphere = left_hemisphere_full[y_min:y_max+1, x_min:x_max+1]
    right_hemisphere = right_hemisphere_full[y_min:y_max+1, x_min:x_max+1]

    # Mirror the right hemisphere
    mirrored_right = np.flip(right_hemisphere, axis=1)

    # Rotate hemispheres for tilt correction
    crop_height, crop_width = left_hemisphere.shape
    center_crop = (crop_width / 2, crop_height / 2)
    M_left = cv2.getRotationMatrix2D(center_crop, angle, 1.0)
    M_right = cv2.getRotationMatrix2D(center_crop, -angle, 1.0)
    rotated_left = cv2.warpAffine(left_hemisphere, M_left, (crop_width, crop_height))
    rotated_mirrored_right = cv2.warpAffine(mirrored_right, M_right, (crop_width, crop_height))

    # Apply thresholding
    rotated_left_thresh = np.where((rotated_left < 175) | (rotated_left > 208), 0, rotated_left)
    rotated_mirrored_right_thresh = np.where((rotated_mirrored_right < 175) | (rotated_mirrored_right > 208), 0, rotated_mirrored_right)

    # Compute mean absolute difference
    diff = np.abs(rotated_left_thresh.astype(np.float32) - rotated_mirrored_right_thresh.astype(np.float32))
    mean_diff = np.mean(diff)

    return mean_diff, skull_region

# ----------------------------------------------------------
# Custom Dataset for CT Images from Google Drive Folders
# ----------------------------------------------------------
class CTImageDataset(Dataset):
    def __init__(self, folder_map, transform=None):
        self.transform = transform
        self.data = []
        binary_label_map = {'ischemia': 0, 'hemorrhagic': 0, 'normal': 1}

        for class_name, folder_path in folder_map.items():
            if not os.path.isdir(folder_path):
                print(f"Folder not found: {folder_path}")
                continue
            for fname in os.listdir(folder_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(folder_path, fname)
                    self.data.append((file_path, binary_label_map.get(class_name, 0)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image_gray is None:
            raise FileNotFoundError(f"Could not load image at {img_path}")

        mean_diff, skull_region = compute_asymmetry_feature(image_gray)
        asym_feature = np.array([mean_diff], dtype=np.float32)

        image_rgb = cv2.cvtColor(skull_region, cv2.COLOR_GRAY2RGB)
        if self.transform:
            image_rgb = self.transform(image_rgb)
        else:
            image_rgb = transforms.ToTensor()(image_rgb)
            image_rgb = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])(image_rgb)
        return image_rgb, torch.tensor(asym_feature), torch.tensor(label, dtype=torch.long)

# ----------------------------------------------------------
# Define Paths to Your Google Drive Folders
# ----------------------------------------------------------
folder_map = {
    'ischemia': '/content/drive/MyDrive/TF_CT/TF_CT_Data_4/train/ischemia',      # Replace with your actual path
    'hemorrhagic': '/content/drive/MyDrive/TF_CT/TF_CT_Data_4/train/bleeding',  # Replace with your actual path
    'normal': '/content/drive/MyDrive/TF_CT/TF_CT_Data_4/train/normal'             # Replace with your actual path
}

# ----------------------------------------------------------
# Define Image Transformations for ResNet50
# ----------------------------------------------------------
data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Create and split dataset
full_dataset = CTImageDataset(folder_map=folder_map, transform=data_transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4, shuffle=False)

# ----------------------------------------------------------
# Define and Train Model
# ----------------------------------------------------------
class ResNetAsymmetryModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetAsymmetryModel, self).__init__()
        self.resnet = models.resnet152(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(num_features + 1, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, asym_feature):
        features = self.resnet(x)
        combined = torch.cat((features, asym_feature), dim=1)
        output = self.classifier(combined)
        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetAsymmetryModel(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Define checkpoint path
checkpoint_path = "/content/drive/MyDrive/TF_CT/checkpoints/best_model_resnet152_simetri_e20.pth"
best_val_acc = 0.0  # Initialize best validation accuracy

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, asym_feature, labels in train_loader:
        images, asym_feature, labels = images.to(device), asym_feature.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images, asym_feature)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    train_loss = running_loss / len(train_dataset)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, asym_feature, labels in val_loader:
            images, asym_feature, labels = images.to(device), asym_feature.to(device), labels.to(device)
            outputs = model(images, asym_feature)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_dataset)
    val_acc = correct / total

    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train Loss: {train_loss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, "
          f"Val Acc: {val_acc:.4f}")

    # Save the model if validation accuracy improves
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Best model saved at epoch {epoch+1} with Val Acc: {val_acc:.4f}")

print("Training complete. Best model saved.")

