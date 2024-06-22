import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tqdm


# Custom dataset class for loading labels and labels
class GraphDataset(Dataset):
    def __init__(self, images_path, labels_path=None, img_size=(600, 600), limit=100, max_boxes=10):
        self.images_path = images_path
        self.labels_path = labels_path
        self.img_size = img_size
        self.limit = limit
        self.max_boxes = max_boxes
        self.image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png'))][:limit]

    def load_yolo_labels(self, label_path):
        labels = []
        with open(label_path, 'r') as file:
            for line in file:
                _, x_center, y_center, width, height = map(float, line.strip().split())
                labels.append([x_center, y_center, width, height])
        return np.array(labels)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        # Load and resize image
        img_path = os.path.join(self.images_path, img_file)
        image = cv2.imread(img_path)
        image = cv2.resize(image, self.img_size)
        image = image / 255.0  # Normalize image data
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        if self.labels_path:
            # Load corresponding label
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(self.labels_path, label_file)

            if os.path.exists(label_path):
                label = self.load_yolo_labels(label_path)
                if label.size == 0:
                    label = np.zeros((1, 4))  # No labels, append zero array
            else:
                label = np.zeros((1, 4))  # No label file, append zero array

            # Pad or truncate labels to the fixed max_boxes length
            if len(label) < self.max_boxes:
                label = np.vstack((label, np.zeros((self.max_boxes - len(label), 4))))
            elif len(label) > self.max_boxes:
                label = label[:self.max_boxes]

            label = torch.tensor(label, dtype=torch.float32)
            return image, label
        else:
            return image


# Define the model
class SimpleCNN(nn.Module):
    def __init__(self, input_shape, num_boxes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (input_shape[0] // 8) * (input_shape[1] // 8), 256)
        self.fc2 = nn.Linear(256, num_boxes * 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = x.view(x.size(0), -1, 4)
        return x


# Load dataset
images_path = "/Users/seanmetlitski/Desktop/objdetect/training_data/images"
labels_path = "/Users/seanmetlitski/Desktop/objdetect/training_data/labels"
img_size = (640, 640)
max_boxes = 10

train_dataset = GraphDataset(images_path, labels_path, img_size, limit=100, max_boxes=max_boxes)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Load validation dataset
val_images_path = "/Users/seanmetlitski/Desktop/objdetect/validation_data/images"
val_labels_path = "/Users/seanmetlitski/Desktop/objdetect/validation_data/labels"

val_dataset = GraphDataset(val_images_path, val_labels_path, img_size, limit=100, max_boxes=max_boxes)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Verify training data loading
print(f"Number of training labels: {len(train_dataset)}")
if train_dataset.labels_path is not None:
    print(f"Number of training labels: {len(train_dataset.image_files)}")
print(f"Number of validation labels: {len(val_dataset)}")

# Get input shape and number of boxes
input_shape = img_size
num_boxes = max_boxes

# Initialize the model, loss function, and optimizer
model = SimpleCNN(input_shape, num_boxes)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Function to calculate Intersection over Union (IoU)
def calculate_iou(box1, box2):
    x1_min, y1_min = box1[:, 0] - box1[:, 2] / 2, box1[:, 1] - box1[:, 3] / 2
    x1_max, y1_max = box1[:, 0] + box1[:, 2] / 2, box1[:, 1] + box1[:, 3] / 2
    x2_min, y2_min = box2[:, 0] - box2[:, 2] / 2, box2[:, 1] - box2[:, 3] / 2
    x2_max, y2_max = box2[:, 0] + box2[:, 2] / 2, box2[:, 1] + box2[:, 3] / 2

    inter_xmin = torch.max(x1_min.unsqueeze(1), x2_min.unsqueeze(0))
    inter_ymin = torch.max(y1_min.unsqueeze(1), y2_min.unsqueeze(0))
    inter_xmax = torch.min(x1_max.unsqueeze(1), x2_max.unsqueeze(0))
    inter_ymax = torch.min(y1_max.unsqueeze(1), y2_max.unsqueeze(0))

    inter_area = torch.clamp(inter_xmax - inter_xmin, min=0) * torch.clamp(inter_ymax - inter_ymin, min=0)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area.unsqueeze(1) + box2_area.unsqueeze(0) - inter_area

    iou = inter_area / union_area

    # Ensure no NaN values in IoU
    iou = torch.nan_to_num(iou, nan=0.0, posinf=0.0, neginf=0.0)

    return iou


# Function to calculate mAP50
def calculate_map50(model, data_loader):
    model.eval()
    all_true_boxes = []
    all_pred_boxes = []

    with torch.no_grad():
        for images, true_boxes in data_loader:
            pred_boxes = model(images)
            all_true_boxes.append(true_boxes)
            all_pred_boxes.append(pred_boxes)

    ious = [calculate_iou(pred, true) for pred, true in zip(all_pred_boxes, all_true_boxes)]

    # Handle potential NaNs
    valid_ious = [iou[~torch.isnan(iou)] for iou in ious]

    # Compute the mean IoU for each sample and then take the mean of these values
    mean_ious = [iou.mean().item() if iou.numel() > 0 else 0 for iou in valid_ious]
    map50 = np.mean(mean_ious)

    return map50


# Training loop with validation and mAP50 calculation
best_map50 = 0.0
map50_scores = []
num_epochs = 75
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation loop
    val_loss = 0.0
    for images, labels in tqdm.tqdm(val_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

    # Calculate mAP50
    map50 = calculate_map50(model, val_loader)
    map50_scores.append(map50)
    if map50 > best_map50:
        best_map50 = map50
        torch.save(model.state_dict(), 'best_models/our_best_model.pth')

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, "
          f"Val Loss: {val_loss / len(val_loader):.4f}, mAP50: {map50:.4f}")

# Save mAP50 scores to a text file
with open('mAP_scores.txt', 'w') as f:
    for score in map50_scores:
        f.write(f"{score}\n")




# Load test dataset and make predictions
# test_images_path = "/Users/seanmetlitski/Desktop/NEW_MODEL/test_img"
#
# test_dataset = GraphDataset(test_images_path, img_size=img_size, max_boxes=max_boxes)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#
# # Verify test data loading
# print(f"Number of test labels: {len(test_dataset)}")
# print(f"Test labels directory: {test_images_path}")
# print(f"Test image files: {os.listdir(test_images_path)}")
#
# if len(test_dataset) == 0:
#     print("No test labels found. Please check the test labels directory and try again.")
# else:
#     model.eval()
#     with torch.no_grad():
#         for i, labels in enumerate(test_loader):
#             outputs = model(labels)
#             print(f"Outputs for Image {i + 1}: {outputs}")
#             outputs = outputs[0].cpu().numpy()
#             img = labels[0].permute(1, 2, 0).cpu().numpy() * 255.0
#             img = img.astype(np.uint8)
#
#             # Ensure image is in correct format
#             img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#
#             # Draw bounding boxes
#             for box in outputs:
#                 x_center, y_center, width, height = box
#                 if x_center == 1.0 and y_center == 1.0 and width == 1.0 and height == 1.0:
#                     print(f"Skipping invalid bounding box: {box}")
#                     continue  # Skip invalid bounding box
#
#                 x_center *= img_size[0]
#                 y_center *= img_size[1]
#                 width *= img_size[0]
#                 height *= img_size[1]
#
#                 x1 = int(x_center - width / 2)
#                 y1 = int(y_center - height / 2)
#                 x2 = int(x_center + width / 2)
#                 y2 = int(y_center + height / 2)
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red color
#
#             # Display the image
#             cv2.imshow(f'Image {i + 1}', img)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
#
#             # Save the image
#             cv2.imwrite(f'output_image_{i + 1}.jpg', img)
