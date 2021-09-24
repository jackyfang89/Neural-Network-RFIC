import numpy as np

"""
Basic Neural Network Procedure:
1. Take in some input dataset
2. Compute some initial weights for input (can be anything, even random)
3. Feed dataset through network of multiplying weights, summing up, then through activation function
4. Calculate error between output of 3. and know good values
5. Adjust weights based on error term using Error Weighted Derivative Formula
6. Repeat steps 3 - 5 for some number of times. 
"""


#basic neural network working with 3x1 input data set. 
class NeuralNetwork():

    #generate initial weights randomly
    def __init__(self):
        np.random.seed(1)
        self.weights = np.random.random((3, 1)) #random weights from [0.0, 1.0)
        self.weights = self.weights * 2 - 1     #normalize to range [-1, 1]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    """
    Helper function for step 3 of procedure. Generates the output of one layer
    input: input dataset to run through network and generate layer output
    """
    def layer_output(self, input):
        input = input.astype(float)
        return self.sigmoid(np.dot(input, self.weights)) #get sum of each input multiplied by weight, then put through sigmoid func

    """
    This function represents step 3 - 5 of NN procedure.
    t_input:  input dataset for training purposes
    t_output: output dataset for training purposes to calculate error term
    t_N:      number of iterations to train NN
    """
    def train(self, t_input, t_output, t_N):
        for i in range(t_N):
            output = self.layer_output(t_input)
            error  = t_output - output
            adjust = np.dot(t_input.T, error * self.sigmoid_derivative(output))
            self.weights += adjust

if __name__ == "__main__":
    NN = NeuralNetwork()
    print("Generate random starting weights: ")
    print(NN.weights)

    t_input = np.array([[0,0,1],   #0
                        [1,1,1],   #1
                        [1,0,1],   #1
                        [0,1,1]])  #0
                        #1, 0, 0
    
    t_output = np.array([[0, 1, 1, 0]]).T #transposed to become vertical

    NN.train(t_input, t_output, 20000)  #train NN 20000 times
    print("Trained weights: ")
    print(NN.weights)

    a = str(input("User Input One: "))
    b = str(input("User Input Two: "))
    c = str(input("User Input Three: "))

    print("New Output: ")
    print(NN.layer_output(np.array([a, b, c])))



  


