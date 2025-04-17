
from torch.optim import Adam
from torch.nn import MSELoss
import torch

from ds import ProvJsonDataset, collate_fn
from model import LSTMForecaster
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

EPOCHS = 150
INITIAL_SAMPLES = 15
FORECAST_HORIZON = 300
TRAIN = True

dataset = ProvJsonDataset()
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

if TRAIN: 
    model = LSTMForecaster()
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = MSELoss()

    model.train()
    losses = []
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch, lengths in dataloader:
            inputs = batch[:, :-1]  # all but last
            targets = batch[:, 1:]  # predict next value

            preds = model(inputs, lengths - 1)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
        losses.append(total_loss)

    torch.save(model, "metric_forecaster/model.pt")

    plt.plot(losses)
    plt.show()

def inverse_diff(diffs, initial=None):
    if initial is None: 
        initial = torch.tensor(diffs[0])
    return torch.cat([initial.unsqueeze(0), initial + torch.cumsum(diffs, dim=0)])

def autoregressive_forecast(model, initial_sequence, forecast_horizon=300):
    model.eval()
    sequence = initial_sequence.clone().detach().tolist()

    with torch.no_grad():
        for _ in range(forecast_horizon):
            input_seq = torch.tensor(sequence[-len(initial_sequence):], dtype=torch.float).unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
            next_val = model.predict_next(input_seq)
            sequence.append(next_val.item())

    return sequence

if not TRAIN: 
    model = LSTMForecaster()
    model.load_state_dict("metric_forecaster/model.pt")

for i in range(5): 
    sample = dataset[i]
    initial_points = sample[:INITIAL_SAMPLES]  # use first 4 points
    ground_truth = sample[INITIAL_SAMPLES:FORECAST_HORIZON+INITIAL_SAMPLES].squeeze()  # next 300 points for reference

    ones = torch.zeros(FORECAST_HORIZON - ground_truth.shape[0])  # creating a vector of 1's using shape of c
    ground_truth = torch.cat((ground_truth, ones), 0)

    predicted = autoregressive_forecast(model, initial_points, forecast_horizon=FORECAST_HORIZON)

    ground_truth = inverse_diff(ground_truth)
    predicted = inverse_diff(predicted)

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(range(INITIAL_SAMPLES), initial_points, label='Initial Input', color='green', marker='o')
    plt.plot(range(INITIAL_SAMPLES, FORECAST_HORIZON+INITIAL_SAMPLES), ground_truth, label='Ground Truth', linestyle='--', color='gray')
    plt.plot(range(INITIAL_SAMPLES, FORECAST_HORIZON+INITIAL_SAMPLES), predicted[INITIAL_SAMPLES:], label='Predicted', color='blue', marker='x')
    plt.title(f'Autoregressive Forecasting ({INITIAL_SAMPLES} → {FORECAST_HORIZON} steps)')
    plt.xlabel('Time step')
    plt.ylabel('Loss value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()