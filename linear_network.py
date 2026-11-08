import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles, make_blobs
from sklearn.preprocessing import OneHotEncoder

X, y = make_circles(n_samples=1000, shuffle=True, noise=0.05, random_state=10, factor=0.8)
#X, y = make_blobs(n_samples=300, centers=4, n_features=2, cluster_std=1.2, random_state=10)
y = y.reshape(-1,1)
g_y = y.copy()
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, y_train.shape)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy(preds, y_true):
    n_samples = y_true.shape[0]
    epsilon = 1e-7
    clipped_preds = np.clip(preds, epsilon, 1. - epsilon)
    logp = -np.log(clipped_preds[np.arange(n_samples), y_true.argmax(axis=1)])
    loss = np.sum(logp) / n_samples
    return loss

def accuracy(y_true, y_pred):
    predictions = np.argmax(y_pred, axis=1)
    targets = np.argmax(y_true, axis=1)
    return np.mean(predictions == targets)*100

class LinearLayer:
    def __init__(self, in_features, out_features):
        self.weight = np.random.randn(in_features, out_features) * 0.01
        self.bias = np.random.randn(1, out_features)
        self.v_weight = np.zeros((in_features, out_features))
        self.v_bias = np.zeros((1, out_features))
    
    def forward(self, input):
        self.input = input
        return np.dot(self.input, self.weight) + self.bias
    
    def backward(self, d_output):
        
        self.d_weight = np.dot(self.input.T, d_output)
        self.d_bias = np.sum(d_output, axis=0, keepdims=True)

        d_input = np.dot(d_output, self.weight.T)
        return d_input
    
    def update(self, lr, beta):
        self.v_weight = beta * self.v_weight + (1 - beta) * lr * self.d_weight
        self.v_bias = beta * self.v_bias + (1 - beta) * lr * self.d_bias

        self.weight -= self.v_weight
        self.bias -= self.v_bias
    
class ReLU:
    def forward(self, x):
        self.input = x
        return np.maximum(0, self.input)
    
    def backward(self, d_output):
        d_input = d_output.copy()
        d_input[self.input <= 0] = 0
        return d_input
    
LL1 = LinearLayer(2,30)
ReLU1 = ReLU()
LL2 = LinearLayer(30,40)
ReLU2 = ReLU()
LL3 = LinearLayer(40,30)
ReLU3 = ReLU()
LL4 = LinearLayer(30,2)

def lr_decay(lr, decay_rate, epoch):
    return lr*(1/(1+decay_rate*epoch))

epochs = 3000
lr_initial = 2
beta = 0.9
decay_rate = 0.1

epoch_plot = []
loss_plot = []

for epoch in range(epochs):
    # --- Forward Pass ---
    out1 = LL1.forward(X_train)
    act1 = ReLU1.forward(out1)
    out2 = LL2.forward(act1)
    act2 = ReLU2.forward(out2)
    out3 = LL3.forward(act2)
    act3 = ReLU3.forward(out3)  
    logits = LL4.forward(act3)
    
    # Final activation
    preds = softmax(logits)

    # --- Loss Calculation ---
    loss = cross_entropy(preds, y_train)
    
    if epoch % 100 == 0:
        epoch_plot.append(epoch)
        loss_plot.append(loss)
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Learning Rate: {lr_decay(lr_initial, decay_rate, epoch):.6f}")

    # --- Backward Pass ---
    # The gradient of Cross-Entropy + Softmax is simply (preds - y_true)
    d_loss = (preds - y_train) / y_train.shape[0]
    d_LL4 = LL4.backward(d_loss)
    d_ReLU3 = ReLU3.backward(d_LL4)
    d_LL3 = LL3.backward(d_ReLU3)
    d_ReLU2 = ReLU2.backward(d_LL3)
    d_LL2 = LL2.backward(d_ReLU2)
    d_ReLU1 = ReLU1.backward(d_LL2)
    d_LL1 = LL1.backward(d_ReLU1)
    
    # --- Update Weights ---
    current_lr = lr_decay(lr_initial, decay_rate, epoch)
    LL4.update(current_lr, beta)
    LL3.update(current_lr, beta)
    LL2.update(current_lr, beta)
    LL1.update(current_lr, beta)

# --- 5. Plotting and Testing ---
plt.figure(figsize=(8, 5))
plt.plot(epoch_plot, loss_plot)
plt.xlabel("Epochs")
plt.ylabel("Cross-Entropy Loss")
plt.title("Loss vs. Epochs")
plt.grid(True)

# Test the model
out1 = LL1.forward(X_test)
act1 = ReLU1.forward(out1)
out2 = LL2.forward(act1)
act2 = ReLU2.forward(out2)
logits = LL3.forward(act2)
test_preds = softmax(logits)

acc = accuracy(test_preds, y_test)
print(f"Test Accuracy: {acc:.2f}%")

def Model(x):
    return softmax(LL4.forward(ReLU3.forward(LL3.forward(ReLU2.forward(LL2.forward(ReLU1.forward(LL1.forward(x))))))))

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
fig, ax = plt.subplots(figsize=(5,5))
i_n = np.c_[xx.ravel(),yy.ravel().T]
z = Model(i_n)
z = np.argmax(z, axis=1)
z = z.reshape(xx.shape)
ax.clear()
plt.scatter(X[:,0], X[:,1], c=g_y, cmap=plt.cm.RdYlBu)
ax.contourf(xx,yy,z,alpha=0.8,cmap=plt.cm.RdYlBu)
plt.show()