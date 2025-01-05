import numpy as np

class SGD_NAG:
    def __init__(self, learning_rate, momentum):
        '''
        Always have an __init__ function in your class, seriously, do it.
        Inputs:
          learning_rate: The step size for the optimization.
          momentum: The momentum coefficient.
        '''

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity_filters = None
        self.velocity_biases = None

    def update(self, filters, biases, filters_grads, biases_grads):
        '''
        After calculating the gradients in the backpass, this function
        will update the parameters using Nesterov Accelerated Gradient.
        Inputs:
          filters: The current filters for the convLayer
          biases: The biases for the convLayer
          filters_grads: The gradients of the filters
          biases_grads: The gradients of the biases
        Output:
          Updated filters and biases.
        '''
        self.filters = filters
        self.biases = biases
        #Initalize the filter and biases velocity
        if self.velocity_filters is None:
            self.velocity_filters = np.zeros_like(filters)
        if self. velocity_biases is None:
            self.velocity_biases = np.zeros_like(biases)

        #lookahead for the filters and biases
        lookahead_filters = self.filters - self.momentum * self.velocity_filters
        lookahead_biases = self.biases - self.momentum * self.velocity_biases

        #update velocities
        self.velocity_filters = self.momentum * self.velocity_filters - self.learning_rate * filters_grads
        self.velocity_biases = self.momentum * self.velocity_biases - self.learning_rate * biases_grads

        #update parameters
        updated_filters = lookahead_filters + self.velocity_filters + self.momentum * (self.velocity_filters - self.momentum * self.velocity_filters)
        updated_biases = lookahead_biases + self.velocity_biases + self.momentum * (self.velocity_biases - self.momentum * self.velocity_biases)

        return updated_filters, updated_biases