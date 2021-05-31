import math
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from matplotlib import pyplot

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        inp, HL1, out  = [1, 10, 1]
        self.linear1 = nn.Linear(inp, HL1)
        self.linear2 = nn.Linear(HL1, out)

    def forward(self, x):
        # normalizing values (periodic function)
        for i in range(len(x)):
            while x[i] < -math.pi:
                x[i] += 2 * math.pi
            while x[i] > math.pi:
                x[i] -= 2 * math.pi

        x = self.linear1(x)
        x = torch.tanh(x)
        x = self.linear2(x)

        return x

class DatasetWrapper(Dataset):
    def __init__(self, x, y):
        xType = torch.FloatTensor
        yType = torch.FloatTensor

        self.length = x.shape[0]
        self.x_data = torch.from_numpy(x).type(xType)
        self.y_data = torch.from_numpy(y).type(yType)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length

def trainBatch(model, x, y, optimizer, loss_fn):
    # Feeding forward
    y_predict = model.forward(x)
    loss = loss_fn(y_predict, y)
    optimizer.zero_grad()

    # Backward propagation
    loss.backward()
    optimizer.step()

    return loss.data.item()

def train(model, loader, optimizer, loss_fn, epochs):
    losses = list()
    batch_index = 0
    for epoch in range(epochs):
        for x, y in loader:
            loss = trainBatch(model=model, x=x, y=y, optimizer=optimizer, loss_fn=loss_fn)
            losses.append(loss)
            batch_index += 1

        print("Epoch: ", epoch + 1)
        print("Batches: ", batch_index)

    return losses

def testBatch(model, x, y):
    # run forward calculation
    y_predict = model.forward(x)
    return y, y_predict

def test(model, loader):
    y_vectors = list()
    y_predict_vectors = list()

    batch_index = 0
    for x, y in loader:
        y, y_predict = testBatch(model=model, x=x, y=y)

        y_vectors.append(y.data.numpy())
        y_predict_vectors.append(y_predict.data.numpy())

        batch_index += 1

    y_predict_vector = np.concatenate(y_predict_vectors)
    return y_predict_vector

def run(datasetTrain, datasetTest, epochs):
    batchSizeTrain = 16
    dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=batchSizeTrain, shuffle=True)
    dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=len(datasetTest), shuffle=False)
    learningRate = 0.05
    network = NeuralNetwork()
    optimizer = optim.SGD(network.parameters(), lr=learningRate)
    loss_fn = nn.L1Loss()  # mean absolute error
    loss = train(model=network, loader=dataLoaderTrain, optimizer=optimizer, loss_fn=loss_fn, epochs=epochs)
    y_predict = test(model=network, loader=dataLoaderTest)

    return loss, y_predict

def taskFunction(x):
    # 302703 Oskar Hacel
    # 274678 Marcin Lisowski
    # p[1] = 3, p[2] = 8 => sin(x * 2) + cos(x * 3)
    return np.sin(x * 2) + np.cos(x * 3)

# training and testing
trainSize = 50000
xTrainData = np.random.uniform(-10, 10, [trainSize, 1])
yTrainData = taskFunction(xTrainData)

xTest = np.random.uniform(-10, 10, [8000, 1])
yTest = taskFunction(xTest)

datasetTrain = DatasetWrapper(x=xTrainData, y=yTrainData)
datasetTest = DatasetWrapper(x=xTest, y=yTest)

print("Train set size: ", datasetTrain.length)
print("Test set size: ", datasetTest.length)

epochs = 4
losses, yPredict = run(datasetTrain=datasetTrain, datasetTest=datasetTest, epochs=epochs)

# plotting the graphs

fig2 = pyplot.figure()
fig2.set_size_inches(10, 5)
pyplot.scatter(xTest, yTest, marker='o', s=0.1)
pyplot.scatter(xTest, yPredict, marker='o', s=0.1)
pyplot.text(-11, 2, "- Prediction", color="orange", fontsize=12)
pyplot.text(-11, 1.8, "- Function", color="blue", fontsize=12)
pyplot.grid()
pyplot.show()

nx = list()

meanY = list()
meanX = list()
tmpMean = 0
for i in range(len(losses)):
    nx.append(i)
    tmpMean += losses[i]
    if i % 125 == 0 and i > 0:
        meanY.append(tmpMean / 125)
        tmpMean = 0
        meanX.append(i)

fig2 = pyplot.figure()
fig2.set_size_inches(10, 5)
pyplot.scatter(nx, losses, marker='o', s=0.1, color="orange")
pyplot.plot(meanX, meanY, color="blue")
pyplot.text(0, 1, "- Loss scatter", color="orange", fontsize=12)
pyplot.text(0, 1.1, "- Average loss", color="blue", fontsize=12)
pyplot.grid()
pyplot.show()