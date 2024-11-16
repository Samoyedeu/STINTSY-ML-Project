import torch.nn as nn
import torch.nn.init


class NeuralNetwork(nn.Module):

    def __init__(self,
             input_size,
             output_size,  # Changed from num_classes
             list_hidden,
             activation='sigmoid'):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.list_hidden = list_hidden
        self.activation = activation

    def create_network(self):
        """Creates the layers of the neural network."""
        layers = []

        # First layer
        layers.append(nn.Linear(self.input_size, self.list_hidden[0]))
        layers.append(self.get_activation())

        # Hidden layers
        for i in range(len(self.list_hidden) - 1):
            layers.append(nn.Linear(self.list_hidden[i], self.list_hidden[i + 1]))
            layers.append(self.get_activation())

        # Output layer (regression output size)
        layers.append(nn.Linear(self.list_hidden[-1], self.output_size))

        self.layers = nn.Sequential(*layers)


    def init_weights(self):
        """Initializes the weights of the network. Weights of a
        torch.nn.Linear layer should be initialized from a normal
        distribution with mean 0 and standard deviation 0.1. Bias terms of a
        torch.nn.Linear layer should be initialized with a constant value of 0.
        """
        torch.manual_seed(2)

        # For each layer in the network
        for module in self.modules():

            # If it is a torch.nn.Linear layer
            if isinstance(module, nn.Linear):

                # TODO: Initialize the weights of the torch.nn.Linear layer
                # from a normal distribution with mean 0 and standard deviation
                # of 0.1.
                # HINT: Use nn.init.normal_() function.
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
                

                # TODO: Initialize the bias terms of the torch.nn.Linear layer
                # with a constant value of 0.
                # HINT: Use nn.init.constant_() function.
                nn.init.constant_(module.bias, 0)
                pass

    def get_activation(self,
                       mode='sigmoid'):
        """Returns the torch.nn layer for the activation function.

        Arguments:
            mode {str, optional} -- Type of activation function. Choices
            include 'sigmoid', 'tanh', and 'relu'.

        Returns:
            torch.nn -- torch.nn layer representing the activation function.
        """
        activation = nn.Sigmoid()

        if mode == 'tanh':
            activation = nn.Tanh()

        elif mode == 'relu':
            activation = nn.ReLU(inplace=True)

        return activation

    def forward_manual(self, x, verbose=False):
        """Manually implemented forward propagation."""
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], nn.Linear):
                x = torch.matmul(x, self.layers[i].weight.T) + self.layers[i].bias
            else:
                x = self.layers[i](x)
            if verbose:
                print(f'Output of layer {i}:\n{x}\n')
        return x  # Direct regression output


    def forward(self, x, verbose=False):
        """Forward propagation of the model, implemented using PyTorch."""
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if verbose:
                print(f'Output of layer {i}:\n{x}\n')
        return x  # Direct regression output


