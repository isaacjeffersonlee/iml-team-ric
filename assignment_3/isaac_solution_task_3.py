"""
Note: this code was copied from a colab notebook so is really messy 🤮
"""
import numpy as np
import torchvision
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

torch.manual_seed(420)
env_path = "."
# weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
# effnet = torchvision.models.efficientnet_b0(weights=weights)  # 20.5MB
weights = torchvision.models.EfficientNet_V2_L_Weights.IMAGENET1K_V1
effnet = torchvision.models.efficientnet_v2_l(weights=weights)
effnet.to(device)
effnet.classifier = nn.Identity()
effnet.eval()

weights.transforms()
# Define the images dataset
img_dataset = torchvision.datasets.ImageFolder(
    root=f"{env_path}/dataset/", transform=weights.transforms()
)
# Define the dataloader for the embeddings
img_loader = DataLoader(
    dataset=img_dataset,
    batch_size=100,
    shuffle=False,
    pin_memory=True,
    num_workers=2,
)
embeddings = []
with torch.no_grad():
    for idx, batch in enumerate(img_loader):
        print(idx)
        batch[0] = batch[0].to(device)
        embeddings.append(effnet(batch[0]))

# List of tensors -> Tensor -> embeddings.pt
torch.save(torch.vstack(embeddings), f"{env_path}/embeddings/embeddings_v2.pt")
# Load and normalize embeddings
embeddings = F.normalize(torch.load(f"{env_path}/embeddings/embeddings_v2.pt"), dim=1)
embeddings = embeddings.to("cpu")
filenames = [s[0].split("/")[-1].rstrip(".jpg") for s in img_dataset.samples]
f_to_emb = {filename: embeddings[i] for i, filename in enumerate(filenames)}
# Training Data
train_triplets = []
with open(f"{env_path}/train_triplets.txt") as f:
    for line in f:
        train_triplets.append(line.rstrip("\n"))
# Split training data into training and validation
train_val_ratio = 0.8
split_idx = int(len(train_triplets) * train_val_ratio)
val_triplets = train_triplets[split_idx:]
train_triplets = train_triplets[:split_idx]
X_train = []
y_train = []
for t in train_triplets:
    emb = [f_to_emb[f] for f in t.split(" ")]
    X_train.append(torch.hstack([emb[0], emb[1], emb[2]]))
    y_train.append(1)
    # Data augmentation
    X_train.append(torch.hstack([emb[0], emb[2], emb[1]]))
    y_train.append(0)

X_train = torch.vstack(X_train)
y_train = torch.tensor(y_train)
print(f"X_train size: {X_train.size()}")
print(f"y_train size: {y_train.size()}")


X_val = []
y_val = []
for t in val_triplets:
    emb = [f_to_emb[f] for f in t.split(" ")]
    X_val.append(torch.hstack([emb[0], emb[1], emb[2]]))
    y_val.append(1)
    # Data augmentation
    X_val.append(torch.hstack([emb[0], emb[2], emb[1]]))
    y_val.append(0)

X_val = torch.vstack(X_val)
y_val = torch.tensor(y_val)
print(f"X_val size: {X_val.size()}")
print(f"y_val size: {y_val.size()}")

# Test Data
test_triplets = []
with open(f"{env_path}/test_triplets.txt") as f:
    for line in f:
        test_triplets.append(line.rstrip("\n"))

X_test = []
y_test = []
for t in test_triplets:
    emb = [f_to_emb[f] for f in t.split(" ")]
    X_test.append(torch.hstack([emb[0], emb[1], emb[2]]))

X_test = torch.vstack(X_test)
print(f"X_test size: {X_test.size()}")

batch_size = 32

emb_train_dataset = TensorDataset(X_train, y_train)
emb_val_dataset = TensorDataset(X_val, y_val)
emb_test_dataset = TensorDataset(X_test)

emb_train_loader = DataLoader(
    dataset=emb_train_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=2,
)

emb_val_loader = DataLoader(
    dataset=emb_val_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=2,
)

emb_test_loader = DataLoader(
    dataset=emb_test_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=2,
)


class IsaacNet(nn.Module):
    def __init__(self):
        super().__init__()
        p = 0.5  # Dropout
        d = 1280 * 3
        self.model = nn.Sequential(
            nn.Linear(in_features=d, out_features=d // 2),
            nn.Dropout(p),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=d // 2),
            nn.Linear(in_features=d // 2, out_features=1),
        )

    def forward(self, x):
        return self.model(x).squeeze(-1).float()


model = IsaacNet()
model.to(device)
model.train()

# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.90)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,  weight_decay=1e-6)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)
loss_fn = torch.nn.BCEWithLogitsLoss()


def train_one_epoch(epoch_index):
    last_loss = 0.0
    for i, data in enumerate(emb_train_loader):
        inputs, labels = data
        labels = labels.to(device)
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels.float())
        loss.backward()
        optimizer.step()
        last_loss = loss.item()
        if i % 12 * batch_size == 0:  # Report loss every 12 batches
            print(f"  batch {i} loss: {last_loss}")

    return last_loss


# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime("%H-%M-%S")
epoch_number = 0
n_epochs = 17
best_vloss = 1_000_000.0
best_model = None
for epoch in range(n_epochs):
    print("EPOCH {}:".format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number)

    # We don't need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0
    for i, vdata in enumerate(emb_val_loader):
        vinputs, vlabels = vdata
        vinputs = vinputs.to(device)
        vlabels = vlabels.to(device)
        voutputs = model(vinputs)
        vloss = loss_fn(voutputs, vlabels.float())
        running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print(f"LOSS train {avg_loss} valid {avg_vloss}")

    # Track best performance on validation set
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        best_model = copy.deepcopy(model)
        # model_path = f"{env_path}/checkpoints/{timestamp}_{epoch_number}_model"
        # torch.save(model.state_dict(), model_path)

    epoch_number += 1

model = best_model
y_val_pred = []
model.eval()
with torch.no_grad():
    for idx, data in enumerate(emb_val_loader):
        inputs, labels = data
        labels = labels.to(device)
        inputs = inputs.to(device)
        outputs = F.sigmoid(model(inputs))
        outputs[outputs > 0.5] = 1.0
        outputs[outputs <= 0.5] = 0.0
        y_val_pred.append(outputs)

y_val_pred = torch.cat(y_val_pred)
y_val_pred.size()

y_val_pred = y_val_pred.to("cpu")
val_accuracy = (y_val_pred == y_val).float().mean()
print(f"Validation Accuracy: {val_accuracy}")

y_test_pred = []
model.eval()
with torch.no_grad():
    for idx, data in enumerate(emb_test_loader):
        inputs = data[0].to(device)
        outputs = F.sigmoid(model(inputs))
        outputs[outputs > 0.5] = 1.0
        outputs[outputs <= 0.5] = 0.0
        y_test_pred.append(outputs)

y_test_pred = torch.cat(y_test_pred)
y_test_pred.size()

# Save Results
predictions = np.array(y_test_pred.to("cpu"))
timestamp = datetime.now().strftime("%H-%M-%S")
np.savetxt(f"{env_path}/results_{timestamp}.txt", predictions, fmt="%i")
print("Done!")
