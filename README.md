# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset
Stock price prediction is an important task in financial analysis because investors and organizations rely on accurate forecasts to make better investment decisions. Traditional statistical methods often struggle to capture complex patterns in time-series data such as stock prices.

The objective of this project is to develop a Recurrent Neural Network (RNN) model that can learn patterns from historical stock price data and predict future prices. Using the historical closing prices of Google stock, the model will be trained on a training dataset and evaluated on a separate test dataset.

The system will involve loading the datasets, preprocessing the data, building and training an RNN model, and then predicting stock prices for the test dataset. Finally, the predicted values will be compared with the actual stock prices to evaluate the performance and accuracy of the model.

### Train Dataset
<img width="560" height="617" alt="image" src="https://github.com/user-attachments/assets/dd313228-5a75-4cbf-a4d2-7eff9958317c" />

### Test Dataset

<img width="558" height="618" alt="image" src="https://github.com/user-attachments/assets/ab7f8d79-b3ba-4113-a83e-f3be8f6c1194" />

## DESIGN STEPS
### STEP 1: 

Load and normalize data, create sequences.

### STEP 2: 

Convert data to tensors and set up DataLoader.

### STEP 3: 

Define the RNN model architecture

### STEP 4: 

Summarize, compile with loss and optimizer.

### STEP 5: 

Train the model with loss tracking.

### STEP 6: 

Predict on test data, plot actual vs. predicted prices.


## PROGRAM

### Name: RAMYA R

### Register Number: 212223230169

```python


## Step 2: Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1,hidden_size=64,num_layers=2,output_size=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_size,output_size)
    def forward(self, x):
        out,_=self.rnn(x)
        out=self.fc(out[:,-1,:])
        return out

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

## Step 3: Train the Model

# Write your code here
def train_model(model,train_loader,criterion,optimizer,epochs=20):
  train_losses=[]
  model.train()
  for epoch in range(epochs):
    total_loss=0
    for x_batch,y_batch in train_loader:
      x_batch,y_batch=x_batch.to(device),y_batch.to(device)
      optimizer.zero_grad()
      outputs=model(x_batch)
      loss=criterion(outputs,y_batch)
      loss.backward()
      optimizer.step()
      total_loss=loss.item()
    train_losses.append(total_loss/len(train_loader))
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

# Plot training loss
  print('Name: RAMYA R')
  print('Register Number: 212223230169')
  plt.plot(train_losses, label='Training Loss')
  plt.xlabel('Epoch')
  plt.ylabel('MSE Loss')
  plt.title('Training Loss Over Epochs')
  plt.legend()
  plt.show()
train_losses = train_model(model,train_loader,criterion,optimizer,epochs=20)
```

### OUTPUT
<img width="782" height="362" alt="image" src="https://github.com/user-attachments/assets/f3b6c7c0-1283-42ef-981d-38f37ac4add3" />

## Training Loss Over Epochs Plot

<img width="730" height="442" alt="image" src="https://github.com/user-attachments/assets/33fa1f82-4e73-4d49-b272-9e95c39b5031" />
<img width="738" height="613" alt="image" src="https://github.com/user-attachments/assets/1b901543-eb0d-4efd-8515-15f0ce11ef69" />


## True Stock Price, Predicted Stock Price vs time

<img width="762" height="517" alt="image" src="https://github.com/user-attachments/assets/1791d504-b708-4032-b73f-f790be7dd034" />


### Predictions
<img width="330" height="57" alt="image" src="https://github.com/user-attachments/assets/ce48b45d-7125-4994-bf06-a4a5a9077c43" />


## RESULT
Thus, a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data has been developed successfully.
