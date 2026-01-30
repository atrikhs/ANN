# Part 1 - Data Preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Load dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

# Encode categorical variables
X = pd.get_dummies(X, columns=['Geography', 'Gender'], drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y.values, test_size=0.2, random_state=0
)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)


y_train = torch.tensor(y_train.reshape(-1,1), dtype=torch.float32)
y_test = torch.tensor(y_test.reshape(-1,1), dtype=torch.float32)

print(X_train.shape)


class AnnModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 6)
        self.layer2 = nn.Linear(6, 6)
        self.output = nn.Linear(6, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # He initialization for layers with ReLU
        nn.init.kaiming_normal_(self.layer1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.layer2.weight, nonlinearity='relu')
        nn.init.zeros_(self.layer1.bias)
        nn.init.zeros_(self.layer2.bias)
        nn.init.zeros_(self.output.bias)


    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x


# Initialize model
input_dim = X_train.shape[1] # no of columns in df  = 11
annmodel = AnnModel(input_dim)
print(annmodel)
print(annmodel.state_dict())


# Loss and optimizer
loss_fn = nn.BCELoss() #nn.L1Loss() #MAE loss
optimizer = optim.SGD(annmodel.parameters(), lr=0.01) #generally between 0.1 to 0.001



# which device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
annmodel.to(device)
# print(next(annmodel.parameters()).device)

# Move data to the same device
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

#Training Loop

torch.manual_seed(42)
epochs = 10000
train_losses = []
test_losses = []

for epoch in range(epochs):
    annmodel.train()

    #Forward pass
    y_preds = annmodel(X_train)

    #Calcuate the Loss
    loss = loss_fn(y_preds, y_train)

    # Optimier Zero Grad. Accumalate gradient by default
    optimizer.zero_grad()

    # perform back prop, calculates gradient wrt loss function for each parameter(weights and bias) in the model
    loss.backward()

    # Optimizer step, uses gradients from above and updates weights w_new = w_eightold - lr(gradient) gradient = dL/dW
    optimizer.step()


    #Testing

    annmodel.eval()

    with torch.inference_mode():
        test_pred = annmodel(X_test)
        test_loss = loss_fn(test_pred,y_test)
        y_preds_class = (test_pred >= 0.5).float()


    train_losses.append(loss.item())
    test_losses.append(test_loss.item())

    if epoch%10 ==0:
        print(f'{epoch =} ,{loss = }, {test_loss = }')


# Plot training and testing loss
plt.figure(figsize=(10,6))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Testing Loss over Epochs')
plt.legend()
plt.show()

# Convert probabilities to class labels (0 or 1)
y_preds_class = (annmodel(X_test) >= 0.5).float()
y_test_np = y_test.cpu().numpy().reshape(-1)         # Flatten to 1D
y_preds_np = y_preds_class.cpu().numpy().reshape(-1) # Flatten to 1D

# Confusion matrix and accuracy
cm = confusion_matrix(y_test_np, y_preds_np)
score = accuracy_score(y_test_np, y_preds_np)

print("Confusion Matrix:\n", cm)
print("Accuracy:", score)


