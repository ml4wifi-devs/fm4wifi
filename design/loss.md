# Loss function for `Enum` with `NA`

Let's have an enum like:

```python
import enum

class C(enum.Enum):
    A: int
    B: int
    NA: int
```

To specify one element we can use `one_hot` encoding. 
`NA` has a special meaning and one could opt for different representation e.g.
a vector of zeros so this becomes _dummy variable_ representation.

How to compute loss from this representation?

Let the network produce `n` outputs $l$ representing all possibilities.
We will kepp ints interpretation as logits.

so 

- $p = p_{NA}=sigmoid(l[-1])$
- $\pi=p_{0:-1}=softmax(l[0:-1])$

The second term has meaning only when the last value is `0` so the distribution under consideration
is a mixture of categorical and deterministic supported at 0.


    P(0) = p
    P(~0) = (1-p)P_{cat}(l[0:-1])
    P(y)=p^y[-1]*((1-p)P_{cat}(l[0:-1]))^(1-y[-1])
    log P(y) = y[-1]log(p) + (1-y[-1])(log(1-p) + logP_{cat}(l[0:-1]))
    log P(y) = y[-1] log_sigmoid(l[-1]) + (1-y[-1]) log(1-sigmoid(l[-1])) + (1-y[-1]) softmax_cross_entropy(l[0:-1], y[0:-1])
    loss = y[-1] sigmoid_cross_entropy(l[-1],y[-1]) + (1-y[-1]) softmax_cross_entropy(l[0:-1], y[0:-1]) + (1-y[-1]) log(1-sigmoid(l[-1]))
    
Now we want to compute `log(1-p)` efficiently

    log(1-sigmoid(l))=log(sigmoid(-l))=-softplus(l)

so the final loss is

    loss = y[-1] sigmoid_cross_entropy(l[-1],y[-1]) + (1-y[-1]) (softmax_cross_entropy(l[0:-1], y[0:-1]) - softplus(l[-1]))
    
