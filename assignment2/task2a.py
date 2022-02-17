import numpy as np
import utils
import typing
from tqdm import tqdm
np.random.seed(1)

def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    # TODO implement this function (Task 2a)
    
    X_train, *_ = utils.load_full_mnist()
    mean = np.mean(X_train).astype(float)
    std = np.std(X_train).astype(float)

    X_norm = (X-mean)/std
    X_pross = np.c_[X_norm, np.ones(X.shape[0])]
    return X_pross


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    # TODO: Implement this function (copy from last assignment)
    cn = -np.sum((targets)*np.log(outputs), axis=1)
    c = np.mean(cn)
    return c


class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        # Always reset random seed before weight init to get comparable results.
        np.random.seed(1)
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid
        self.use_improved_weight_init = use_improved_weight_init

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        self.ws = []        
        self.zs= []

        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            #w = np.zeros(w_shape)
            if use_improved_weight_init == True:                    # Task 3a
                w = np.random.normal(0, 1 / np.sqrt(prev), w_shape) # 0 mean and 1/785 standard deviation
            else:
                w = np.random.uniform(-1, 1, w_shape)               # Task 2c
            self.ws.append(w)
            prev = size
        self.grads = []

    def softmax(self, Z):
        Y_hat = np.exp(Z)/np.sum(np.exp(Z), axis=1, keepdims=True)
        return Y_hat

    def sigmoid(self, Z):
        if self.use_improved_sigmoid == True:
            Z_sigd = 1.7159 * np.tanh((2/3) * Z) #Task 3b
        else:
            Z_sigd = 1/(1+np.exp(-Z))
        return Z_sigd

    def dsigmoid(self, Z):
        if self.use_improved_sigmoid == True:
            Z_sigd = 2.28787 / (np.cosh((4/3) *Z) + 1)
        else:
            Z_sigd = self.sigmoid(Z)*(1-self.sigmoid(Z))
        return Z_sigd

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # TODO implement this function (Task 2b)
        # HINT: For performing the backward pass, you can save intermediate activations in variables in the forward pass.
        # such as self.hidden_layer_output = ...
        
        # Task 2
        """
        A0 = X
        Z0 = A0 @ self.ws[0]
        A1 = self.sigmoid(Z0)
        Z1 = A1 @ self.ws[-1]
        Y_hat = self.softmax(Z1)
        self.hidden_layer_output = [A0, A1, Y_hat]
        self.zs = [Z0, Z1]
        """
        # Task 4
        self.zs = []
        A = X
        self.hidden_layer_output = [A]

        for i in range(len(self.neurons_per_layer)-1):
            Z = A @ self.ws[i]
            A = self.sigmoid(Z)
            self.zs.append(Z)
            self.hidden_layer_output.append(A)
        Z_final = A @ self.ws[-1]
        Y_hat = self.softmax(Z_final)
        self.zs.append(Z_final)
        self.hidden_layer_output.append(Y_hat)
        
        return Y_hat

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad
        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 2b)

        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
        
        #Both 2 and 4
        self.grads = [np.zeros_like(w) for w in self.ws]
        N = X.shape[0]
        dirac_k = -(targets-outputs) 
        self.grads[-1] = (self.hidden_layer_output[-2].T @ dirac_k) / N

        # Task 2
        """
        df = self.dsigmoid(self.zs[-2])
        dirac_j = df * (dirac_k @ self.ws[-1].T)
        self.grads[-2] = (X.T @ dirac_j) / N
        """
        # Task 4
        dirac_j = dirac_k
        for i in range(len(self.neurons_per_layer)-1):
            df = self.dsigmoid(self.zs[-i-2])
            dirac_j = df * (dirac_j @ self.ws[-i-1].T)
            self.grads[-i-2] = (self.hidden_layer_output[-i-3].T @ dirac_j) / N
        # End task 4

        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."
        
    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    # TODO: Implement this function (copy from last assignment)
    Y_hot_one = np.zeros((Y.shape[0],num_classes))
    for i in range(Y.shape[0]):
        Y_hot_one[i,Y[i]] = 1
    return Y_hot_one


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in (enumerate(model.ws)):
        for i in tqdm(range(w.shape[0])):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
            
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"

        
if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64,10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)