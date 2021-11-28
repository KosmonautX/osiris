import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.nn.utils import weight_norm
from sklearn.metrics import mean_squared_error, mean_absolute_error

def endtoend(target, cols, raw, sector):
    #--------------------------------------------------------------BOLLINGER BAND---------------------------------------
    raw.groupby([pd.Grouper(key="time", freq="Y")])[target].sum()
    raw["total"] = raw[target]
    raw["sd"] = raw["total"].rolling(16).std(ddof=0)
    raw["mean"] = raw["total"].rolling(16).mean()
    raw['bolu'] = raw['mean'] + 2*raw['sd']
    raw['bold'] = raw['mean'] - 2*raw['sd']

    #--------------------------------------------------------------INITIAL CELL---------------------------------------
    train_length = int(0.8*len(raw))
    depth = 4
    batch_size = 32
    prediction_horizon = 1

    train = raw.iloc[:train_length, :]
    valid = raw.iloc[train_length:, :]

    #X = np.zeros((len(raw), depth, len(cols)))
    #for i, name in enumerate(cols):
    #    for j in range(depth):
    #        X[:, j, i] = raw[name].shift(depth - j - 1).bfill()
    #y = raw[target].shift(-1).ffill().values
    X = np.zeros((len(train), depth, len(cols)))
    for i, name in enumerate(cols):
        for j in range(depth):
            X[:, j, i] = train[name].shift(depth - j - 1).bfill()
    Y = train[target].shift(-1).ffill().values

    x = np.zeros((len(valid), depth, len(cols)))
    for i, name in enumerate(cols):
        for j in range(depth):
            x[:, j, i] = valid[name].shift(depth - j - 1).bfill()
    y = valid[target].shift(-1).ffill().values

    #--------------------------------------------------------------INITIAL CELL---------------------------------------
    X_train, X_train_max , X_train_min = relative(X)
    X_test, X_test_max , X_test_min = relative(x)

    y_train, y_train_max , y_train_min = relative(Y)
    y_test, y_test_max , y_test_min = relative(y)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    #-----------------------------------------------------------TRAINING----------------------------------------------
    model = TCN(X_train.shape[2], 5).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=0.002)
    sc = torch.optim.lr_scheduler.StepLR(opt, 10, 0.9)
    loss = nn.MSELoss()
    early_stopping_rounds=10
    vl = 99999
    counter = 0
    for e in range(300):
        train_loss = 0
        val_loss = 0
        preds = []
        true = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()

            opt.zero_grad()

            output = model(batch_x)
            true.append(batch_y.detach().cpu().numpy())
            preds.append(output.detach().cpu().numpy())

            l = loss(output, batch_y)

            l.backward()

            opt.step()

            train_loss += l.item()
        true = np.concatenate(true)
        preds = np.concatenate(preds)

        sc.step()
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()

                output = model(batch_x)
                l = loss(output, batch_y)
                val_loss += l.item()


        if vl > val_loss:
            vl = val_loss
            torch.save(model.state_dict(), 'birth_' + sector+ '.pt')
            counter = 0

        else:
            counter += 1

        if counter >= early_stopping_rounds:
            break

        if (e%10 == 0):
            preds = preds*(y_train_max - y_train_min) + y_train_min
            true = true*(y_train_max - y_train_min) + y_train_min
            mse = mean_squared_error(true, preds)
            mae = mean_absolute_error(true, preds)
            print("mse: ", mse, "mae: ", mae)
            plt.figure(figsize=(20, 10))
            plt.plot(preds, label='Predicted Birth Rates against' + sector)
            plt.plot(true)
            plt.xlabel("Training Timesteps")
            plt.ylabel("Total Live Births")
            plt.legend(loc="upper right")
            plt.show()

        print('Iter: ', e, 'train_loss: ', train_loss, 'val_loss: ', val_loss)

    preds = []
    true = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()

            output = model(batch_x)

            true.append(batch_y.detach().cpu().numpy())
            preds.append(output.detach().cpu().numpy())

    true = np.concatenate(true)
    preds = np.concatenate(preds)

    raw.loc[len(raw)-len(preds):len(raw), 'pred'] = preds*(y_test_max-y_test_min+ 1e-9) + y_test_min
    raw.loc[len(raw)-len(preds):len(raw), 'test'] = true*(y_test_max-y_test_min+ 1e-9) + y_test_min
    plt.style.use('dark_background')
    plt.plot(raw['bolu'],color="red", label='Upper Bollinger Band')
    plt.plot(raw['bold'],color="yellow", label='Lower Bollinger Band')
    plt.plot(raw['total'], label='True Validation Data')
    plt.plot(raw['pred'],color="blue", label='Predicted Birth Rates')
    plt.title("live birth vs " + sector )
    plt.xlabel("Validation Timesteps")
    plt.ylabel("Total Live Births")
    plt.legend(loc="upper right")
    plt.fill_between(raw.index,raw['bolu'],raw['bold'], color='gray')
    plt.figure(figsize=(10, 10), dpi=1200)
    plt.savefig('plot_outputs/birthvs' + sector +'.png')
    mse = mean_squared_error(true, preds)
    mae = mean_absolute_error(true, preds)
    #classification_report(true,preds)
    print(mse, mae)


def relative(data):
    return torch.Tensor((data-data.min(axis=0))/(data.max(axis=0)-data.min(axis=0)+ 1e-9)), data.max(axis=0), data.min(axis=0)

class TCNTemporalBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dilation):
        super().__init__()
        padding = int(dilation*(kernel_size-1))
        self.pad = nn.ConstantPad1d((padding, 0), 0)
        self.conv1 = weight_norm(nn.Conv1d(input_channels, output_channels, kernel_size, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(output_channels, output_channels, kernel_size, dilation=dilation))
        self.residual = nn.Conv1d(input_channels, output_channels, 1)

    def forward(self, x):
        out = self.pad(x)
        out = torch.relu(self.conv1(out))
        out = self.pad(out)
        out = torch.relu(self.conv2(out))
        y = self.residual(x)
        out = torch.relu(out + y)
        return out


class TCN(nn.Module):
    def __init__(self, input_dim, n_layers, n_channels=32, kernel_size=3):
        super().__init__()
        self.first_layer = TCNTemporalBlock(input_dim, n_channels, kernel_size, 1)
        self.tcn_layers = nn.ModuleList([TCNTemporalBlock(n_channels, n_channels, kernel_size, 2**(i+1)) for i in range(n_layers-1)])
        self.n_layers = n_layers
        self.output_transform = nn.Linear(n_channels, 1)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.first_layer(x)
        for i in range(self.n_layers-1):
            x = self.tcn_layers[i](x)
        x = x[..., -1]
        out = self.output_transform(x)
        return out.squeeze(1)
