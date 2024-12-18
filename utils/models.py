import torch
import torch.nn as nn
import torch.nn.functional as F

class ForwardCNN(nn.Module):
    def __init__(self, input_layer_size, output_layer_size):
        super(ForwardCNN, self).__init__()

        self.pool2 = nn.MaxPool1d(2)
        self.pool3 = nn.MaxPool1d(3)

        self.dropout = nn.Dropout(p=0.5)

        self.conv1t = nn.ConvTranspose1d(input_layer_size, int(4 * output_layer_size / 3), kernel_size=40, stride=3)
        self.conv1 = nn.Conv1d(int(4 * output_layer_size / 3), int(output_layer_size), kernel_size=22)
        self.conv2 = nn.Conv1d(int(output_layer_size), int(output_layer_size), kernel_size=19)

        self.conv3t = nn.ConvTranspose1d(int(output_layer_size), int(3*output_layer_size/2), kernel_size=40, stride=4)
        self.conv3 = nn.Conv1d(int(3*output_layer_size/2), int(4 * output_layer_size / 3), kernel_size=16)
        self.conv4 = nn.Conv1d(int(4 * output_layer_size / 3), int(4 * output_layer_size / 3), kernel_size=5)

        self.conv5t = nn.ConvTranspose1d(int(4 * output_layer_size / 3), int(2*output_layer_size), kernel_size=21, stride=4)#4 2
        self.conv5 = nn.Conv1d(int(2*output_layer_size), int(output_layer_size), kernel_size=15) #15 32
        self.final_conv = nn.Conv1d(int(output_layer_size), int(output_layer_size), kernel_size=5) #5 30

    def forward(self, x):
        # print("Input:", x.shape)
        x = F.leaky_relu(self.conv1t(x), 0.1)
        x = self.dropout(x)
        # print("conv1t:", x.shape)
        x = F.leaky_relu(self.conv1(x), 0.1)
        # print("conv1:", x.shape)
        x = F.leaky_relu(self.conv2(x), 0.1)
        # print("conv2:", x.shape)
        x = self.dropout(x)

        x = F.leaky_relu(self.conv3t(x), 0.1)
        x = self.dropout(x)
        # print("conv3t:", x.shape)
        x = self.pool2(x)
        # print("pool1:", x.shape)
        x = F.leaky_relu(self.conv3(x), 0.1)
        # print("conv3:", x.shape)
        x = F.leaky_relu(self.conv4(x), 0.1)
        # print("conv4:", x.shape)
        x = self.dropout(x)

        x = F.leaky_relu(self.conv5t(x), 0.1)
        # print("conv5t:", x.shape)
        x = F.leaky_relu(self.conv5(x), 0.1)
        # print("conv5:", x.shape)
        x = self.final_conv(x)
        # print("final_conv:", x.shape)
        x = self.pool3(x)
        # print("pool3:", x.shape)

        x = torch.flatten(x, 1)
        # print("flatten:", x.shape)

        return x


class InverseCNN(nn.Module):
    def __init__(self, input_layer_size, output_layer_size):
        super(InverseCNN, self).__init__()

        self.pool2 = nn.MaxPool1d(2)
        self.pool3 = nn.MaxPool1d(3)

        self.dropout = nn.Dropout(p=0.5)

        self.conv1t = nn.ConvTranspose1d(input_layer_size, int(2 * input_layer_size), kernel_size=40, stride=3)
        self.conv1 = nn.Conv1d(int(2 * input_layer_size), int(2 * input_layer_size), kernel_size=22)
        self.conv2 = nn.Conv1d(int(2 * input_layer_size), int(input_layer_size), kernel_size=19)

        self.conv3t = nn.ConvTranspose1d(int(input_layer_size), int(input_layer_size), kernel_size=40, stride=4)
        self.conv3 = nn.Conv1d(int(input_layer_size), int(input_layer_size / 2), kernel_size=16)
        self.conv4 = nn.Conv1d(int(input_layer_size / 2), int(input_layer_size / 2), kernel_size=5)

        self.conv5t = nn.ConvTranspose1d(int(input_layer_size / 2), int(input_layer_size / 3), kernel_size=21,
                                         stride=4)  # 4 2
        self.conv5 = nn.Conv1d(int(input_layer_size / 3), int(output_layer_size), kernel_size=15)  # 15 32
        self.final_conv = nn.Conv1d(int(output_layer_size), int(output_layer_size), kernel_size=5)  # 5 30

    def forward(self, x):
        # # print("Input:", x.shape)
        x = F.leaky_relu(self.conv1t(x), 0.1)
        # print("conv1t:", x.shape)
        x = F.leaky_relu(self.conv1(x), 0.1)
        # print("conv1:", x.shape)
        x = F.leaky_relu(self.conv2(x), 0.1)
        # print("conv2:", x.shape)
        x = self.dropout(x)

        x = F.leaky_relu(self.conv3t(x), 0.1)
        # print("conv3t:", x.shape)
        x = self.pool2(x)
        # print("pool1:", x.shape)
        x = F.leaky_relu(self.conv3(x), 0.1)
        # print("conv3:", x.shape)
        x = F.leaky_relu(self.conv4(x), 0.1)
        # print("conv4:", x.shape)
        x = self.dropout(x)

        x = F.leaky_relu(self.conv5t(x), 0.1)
        # print("conv5t:", x.shape)
        x = F.leaky_relu(self.conv5(x), 0.1)
        # print("conv5:", x.shape)
        x = F.leaky_relu(self.final_conv(x), 0.1)
        # print("final_conv:", x.shape)
        x = self.pool3(x)
        # print("pool3:", x.shape)

        x = torch.flatten(x, 1)
        # print("flatten:", x.shape)

        return x
class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        # Initialize the output with zeros
        ensemble_output = torch.zeros_like(self.models[0](x))

        # Iterate through the models and accumulate their outputs
        for model in self.models:
            ensemble_output += model(x)

        # Divide by the number of models to get the average
        ensemble_output /= len(self.models)

        return ensemble_output


