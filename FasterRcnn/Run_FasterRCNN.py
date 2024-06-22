from FasterRCNN import *
from torch.utils.data import DataLoader

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("device: " + str(device))


def collate_fn(batch):
    return tuple(zip(*batch))


# Initialize the model
model = get_model(pretrained=True)
model = model.to(device)

# Use our dataset and defined transformations
pwd = os.getcwd()
training_dataset = CustomDataset(pwd + r'/new_training_data', transforms=transforms.Compose([transforms.ToTensor()]))
val_dataset = CustomDataset(pwd + r'/validation_data', transforms=transforms.Compose([transforms.ToTensor()]))

train_loader = DataLoader(training_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

lr = 0.0001
# Define an optimizer and a learning rate scheduler
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
# Training loop
num_epochs = 50
train_losses = []
val_losses = []
map_scores = []
best_mAP = 0
for epoch in range(num_epochs):
    model.to(device)
    model.train()
    for images, targets in tqdm(train_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        train_losses.append(losses.item())

    #mAP, avg_loss, avg_classification_loss, avg_box_regression_loss = evaluate_and_get_losses(model, val_loader)
    mAP, avg_loss, avg_classification_loss, avg_box_regression_loss = evaluate_faster_model(model, val_loader)
    map_scores.append(mAP)
    val_losses.append(avg_loss)
    if mAP > best_mAP:
        best_mAP = mAP
        print("Saving the best model with mAP: ", best_mAP)
        if not os.path.exists('models'):
            os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), os.path.join(pwd, 'models/faster_rcnn_complex_pretrained.pth'))

    print(
        f"Epoch #{epoch} MAP: {map_scores[-1]} Loss: {avg_loss} Classification Loss: {avg_classification_loss} Box Regression Loss: {avg_box_regression_loss}")


# save all the losses and map scores at faster_rcnn_runs
if not os.path.exists('faster_rcnn_runs'):
    os.makedirs('faster_rcnn_runs', exist_ok=True)

with open('faster_rcnn_runs/complex_not_pretrained', 'w') as f:
    f.write("Train Losses: " + str(train_losses) + "\n")
    f.write("Validation Losses: " + str(val_losses) + "\n")
    f.write("mAP Scores: " + str(map_scores) + "\n")
    f.write("Best Val mAP: " + str(best_mAP) + "\n")
    f.write("optimizer: " + str(optimizer) + "\n")
    f.write("lr: " + str(lr) + "\n")
    f.write("num_epochs: " + str(num_epochs) + "\n")
    f.close()
print("Training complete")
