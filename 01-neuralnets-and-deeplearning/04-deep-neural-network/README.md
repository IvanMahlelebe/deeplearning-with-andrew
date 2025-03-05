# <b>A Multi-layer Perceptron</b>

In our previous exercise, we implemented a neural network with $3$ layers i.e. one input layer, one hidden layer and one output layer. In this exercise, we're going to generalise to L-layers so that we can be able to scale it in any way we want. As we've done before, the workflow of buildign a neural is going to be the similar(with a few additions):

* [x] Initialise parameters for an $L$-layer network
* [ ] Implement forward propagation
* [ ] Compute the loss
* [ ] Implement backward propagation
* [ ] Update parameters

***NB:** The architecture I've described above is sometimes referred to as a $2$-layer network, disregarding the input layer. Which makes sense because they $L$ in this case refers to the number of ***learnable*** parameters. For now however, we'll stick to $L$ referring to all layers to avoid confusion.*

## Initialisation

The initialisation component is composed of $2$ sub-components:

* Initialise the architecture of the model i.e. number of layers and the number of units in each layer
* Initialise the parameters of the model

In our $3$-layer model in our previous exercise, we initialised the architecture with the tuple $(2, 4, 1)$, meaning we had $3$ layers, each with $2$, $4$ and $1$ units respectively. For simplicity, let's use the same approach for $L$ layers... *seems flexible to me,* I don't have any more solid reason for this.

The part where you have to pay most attention in this function, is the part where you have to decide on the dimensions of both $W$ and $b$. The dimensions of $b$ are rather easy because it has to be a column vector that corresponds to the number of neurons/units in each layer. Therefore, with the previous example, we'll have: $b^{[0]}\in\mathbb{R}^{2}, b^{[1]}\in\mathbb{R}^{4},\text{ and } b^{[2]}\in\mathbb{R}^{1}$.

>In general, if `layer_dims`=$\{n_l\}^{L}$, then $b^{[l]}\in\mathbb{R}^{n_l}$ where $n_l$ is the number of units in layer $l$.

Determining the dimensions of $W$ is also easy, but requires a little more thought. Peeking into the next step, we know that we're going to compute $Z$ as a linear function where $W$ has to multiply $X$. Here are the two points to consider:

* $X\in\mathbb{R}^{n_x\times m}$, where $n_x$ is the number of features and $m$ is the number of training examples in the batch
* Since $Z$ has to connect all features of $X$ with all the units/entries of the first hidden layer, then it must be that $Z\in\mathbb{R}^{n_x\times n_l}$
* $W$ will therefore be the bridge between $X$ and $Z$, and in order for $Z$ to have the above-mentioned dimensions, then it must be that $W\in\mathbb{R}^{m\times n_l}$

In a more general sense, we can expand this logic throughout the network, all the way to the output layer. This is made simple if we can call inputs $A$ such that $Z^{[l]}=W^{[l]}\cdot A^{(l-1)} + b^{[l]}$ where $A^{[0]}=X$, you can thus apply the above logic by induction.

> Anyway, the point here is that $W$ must be of dimensions $(n_l, n_{l-1})$ where $n_{l-1}$ is the number of units(rows) in the previous layer.

**Side note:** Indeed $b$ is a column vector being added to a multi-dimensional array. This doesn't happen in Linear Algebra as addition is element-wise, but this operation is possible in Python through `broadcasting`. This is why it's imperative that you keep it as $b\in\mathbb{R}^{n_l\times 1}$ instead of just $b\in\mathbb{R}^{n_l}$.

*I also might just be missing something here... Back in school, I remember having to transpose $X$ before multiplying it with $W$ so that the matrix multiplication can work, but I can't seem to recall the exact context of when that had to happen.*

## Forward Propagation

The forward propagation computation is composed of the following "sub" computations: 

  * $Z^{[l]}=W^{[l]}\cdot A^{(l-1)} + b^{[l]}$ where $A^{[0]}=X$ as we've discussed above
  * $A^{[l]}=g^{[l]}(Z^{[l]})$ where $g^{[l]}$ is the activation function of layer $l$