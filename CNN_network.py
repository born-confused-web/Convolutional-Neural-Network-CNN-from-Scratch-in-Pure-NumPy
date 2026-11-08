import numpy as cp
import cupy as cp
print(cp.cuda.runtime.getDeviceCount())
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_openml

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

# Extract features and labels
X, y = mnist['data'], mnist['target']

X=X.reshape(70000,28,28)

y = y.reshape(-1, 1)

X = cp.array(X)

encoder = OneHotEncoder(sparse_output=False)

# Fit and transform
y_encoded = encoder.fit_transform(y)

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

#imp classes
class ConvoLayer:
    def __init__(self, k_size, k_depth=None):
        if k_depth:
            self.k1 = cp.random.rand(k_depth,k_size, k_size)
            self.k2 = cp.random.rand(k_depth,k_size, k_size)
        else:
            self.k1 = cp.random.rand(k_size, k_size)
            self.k2 = cp.random.rand(k_size, k_size)
        self.v_k1 = cp.zeros_like(self.k1)
        self.v_k2 = cp.zeros_like(self.k2)
    
    def convolve1(self, X, k):
        w,h = k.shape
        W,H = X.shape[1], X.shape[2]
        out = []
        ele = cp.zeros((W-w+1,H-h+1))
        for x in X:
            for i in range(W-w+1):
                for j in range(H-h+1):
                    ele[i,j] = float(cp.sum(x[i:i+w,j:j+h]*k))
            out.append(ele.copy())
        return cp.array(out)

    def convolve2(self, X, k):
        C,W,H = X.shape[1], X.shape[2], X.shape[3]
        w,h = k.shape[1], k.shape[2]
        out = []
        ele = cp.zeros((W-w+1,H-h+1))
        for x in X:
            for i in range(W-w+1):
                for j in range(H-h+1):
                    ele[i,j] = float(cp.sum(x[:,i:i+w,j:j+h]*k))
            out.append(ele.copy())
        return cp.array(out)

    def forward(self, X):
        self.input = X
        l = X.shape
        if len(l) == 3:
            return cp.stack((self.convolve1(X, self.k1), self.convolve1(X, self.k2)),axis=1)
        elif len(l) == 4:
            return cp.stack((self.convolve2(X, self.k1), self.convolve2(X, self.k2)),axis=1)
            
    def backward(self, d_output):
        self.d_k1 = cp.zeros_like(self.k1)
        self.d_k2 = cp.zeros_like(self.k2)
        if len(self.input.shape) == 3:
            for i in range(d_output.shape[0]):
                self.d_k1 += float(cp.sum(self.convolve1(self.input[i:i+1], d_output[i,0])))
                self.d_k2 += float(cp.sum(self.convolve1(self.input[i:i+1], d_output[i,1])))
        
        elif len(self.input.shape) == 4:
            for i in range(d_output.shape[0]):
                self.d_k1 += cp.stack((self.convolve1(self.input[i:i+1,0], d_output[i,0]),self.convolve1(self.input[i:i+1,1], d_output[i,0])),axis=1).squeeze()
                self.d_k2 += cp.stack((self.convolve1(self.input[i:i+1,0], d_output[i,1]),self.convolve1(self.input[i:i+1,1], d_output[i,1])),axis=1).squeeze()
            padding = self.k1.shape[1] - 1
            d_input = cp.stack([self.convolve2(cp.stack([cp.pad(d_output[:,0],[(0,0),(padding,padding),(padding,padding)])]*2,axis=1), cp.flip(self.k1, (1,0))), self.convolve2(cp.stack([cp.pad(d_output[:,1],[(0,0),(padding,padding),(padding,padding)])]*2,axis=1), cp.flip(self.k2, (0,1)))], axis=1)
            return d_input
    
    def optimize(self, lr, beta):
        self.v_k1 = beta * self.v_k1 + (1 - beta) * lr * self.d_k1
        self.v_k2 = beta * self.v_k2 + (1 - beta) * lr * self.d_k2

        self.k1 -= self.v_k1
        self.k2 -= self.v_k2

class Activation:
    def __init__(self, activation_type=None):
        self.activation_type = activation_type
    
    def sigmoid(self, x):
        return 1/(1+cp.exp(-x))

    def d_sigmoid(self, x):
        return self.sigmoid(x) * (1-self.sigmoid(x))

    def ReLU(self, x):
        return x * (x > 0)

    def d_ReLU(self, x):
        return (x>0)*cp.ones(x.shape)
    
    def forward(self, X):
        self.input = X
        if self.activation_type == 'sigmoid':
            return self.sigmoid(X)

        elif self.activation_type == 'relu':
            return self.ReLU(X)
        
        else:
            return X
    
    def backward(self, d_output):
        if self.activaiton_type=='sigmoid':
            return self.d_sigmoid(self.input)*d_output
        
        elif self.activation_type=='relu':
            return self.d_ReLU(self.input)*d_output

class LinearLayer:
    def __init__(self, in_features, out_features):
        self.weight = cp.random.randn(in_features, out_features) * 0.01
        self.bias = cp.random.randn(1, out_features)
        self.v_weight = cp.zeros((in_features, out_features))
        self.v_bias = cp.zeros((1, out_features))
    
    def forward(self, input):
        self.input = input
        return cp.dot(self.input, self.weight) + self.bias
    
    def backward(self, d_output):
        
        self.d_weight = cp.dot(self.input.T, d_output)
        self.d_bias = float(cp.sum(d_output, axis=0, keepdims=True))

        d_input = cp.dot(d_output, self.weight.T)
        return d_input
    
    def update(self, lr, beta):
        self.v_weight = beta * self.v_weight + (1 - beta) * lr * self.d_weight
        self.v_bias = beta * self.v_bias + (1 - beta) * lr * self.d_bias

        self.weight -= self.v_weight
        self.bias -= self.v_bias
    
class PoolingLayer:
    def __init__(self):
        pass

    def forward(self,X):
        self.input=X
        a,b,c,d = X.shape
        output = cp.zeros((a,b,int(c/2),int(d/2)))
        for n in range(a):
            for c in range(b):
                for i in range(0,c,2):
                    for j in range(0,d,2):
                        output[n,c,i//2,j//2] = cp.max(X[n,c,i:i+2,j:j+2])
        return output

    def backward(self,d_output):
        d_input = cp.zeros_like(self.input)
        a,b,c,d = d_output.shape
        for n in range(a):
            for c in range(b):
                for i in range(c):
                    for j in range(d):
                        patch = self.input[n,c,2*i:2*i+2,2*j:2*j+2]
                        max_val = cp.max(patch)
                        for m in range(2):
                            for l in range(2):
                                if patch[m,l] == max_val:
                                    d_input[n,c,2*i+m,2*j+l] += d_output[n,c,i,j]
        return d_input

def lr_decay(lr, decay_rate, epoch):
    return lr*(1/(1+decay_rate*epoch))

def softmax(x):
    exp_x = cp.exp(x - cp.max(x, axis=1, keepdims=True))
    return exp_x / cp.sum(exp_x, axis=1, keepdims=True)

def cross_entropy(preds, y_true):
    n_samples = y_true.shape[0]
    epsilon = 1e-7
    clipped_preds = cp.clip(preds, epsilon, 1. - epsilon)
    logp = -cp.log(clipped_preds[cp.arange(n_samples), y_true.argmax(axis=1)])
    loss = float(cp.sum(logp) / n_samples)
    return loss

def accuracy(y_true, y_pred):
    predictions = cp.argmax(y_pred, axis=1)
    targets = cp.argmax(y_true, axis=1)
    return cp.mean(predictions == targets)*100

#define model
test = cp.random.rand(10,28,28)
convo1 = ConvoLayer(5)
act1 = Activation('sigmoid')
pool1 = PoolingLayer()
convo2 = ConvoLayer(3,2)
act2 = Activation('sigmoid')
pool2 = PoolingLayer()
LL1 = LinearLayer(50,75)
act3 = Activation('relu')
LL2 = LinearLayer(75,75)
act4 = Activation('relu')
LL3 = LinearLayer(75,10)

#training loop

epochs = 5
lr_initial = 2
beta = 0.9
decay_rate = 0.1

epoch_plot = []
loss_plot = []
for epoch in range(epochs):
    #forward pass
    h_preds = cp.resize((
                        pool2.forward(
                            act2.forward(
                                convo2.forward(
                                    pool1.forward(
                                        act1.forward(
                                            convo1.forward(
                                                X_train[0:1]))))))),(1,50))
    preds = softmax(
        LL3.forward(
            act4.forward(
                LL2.forward(
                    act3.forward(
                        LL1.forward(
                            h_preds))))))







