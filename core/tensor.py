import numpy as np
from graphviz import Digraph

class Tensor:
    def __init__(self, data, _children=(), _op='', label=None):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')
        def _backward():
            # z = x + y, dz/dx = 1, dz/dy = 1
            # 把 out 的梯度原样分配给两个输入节点
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')
        def _backward():
            # z = x * y, dz/dx = y, dz/dy = x
            # 按链式法则乘以上游梯度 out.grad
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor((self.data > 0) * self.data, (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        out = Tensor(np.tanh(self.data), (self,), 'tanh')
        def _backward():
            self.grad += (1 - np.tanh(self.data) ** 2) * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1


    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1000.0
        for v in reversed(topo):
            v._backward()

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
    
    for n in nodes:
        dot.node(name=str(id(n)), label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot

# x0 = Tensor(1.0, label='x0')
# w0 = Tensor(2.0, label='w0')
# x1 = Tensor(3.0, label='x1')
# w1 = Tensor(4.0, label='w1')
# b = Tensor(-7.0, label='b')
# x0w0 = x0 * w0; x0w0.label = 'x0w0'
# x1w1 = x1 * w1; x1w1.label = 'x1w1'
# x0w0x1w1 = x0w0 + x1w1; x0w0x1w1.label = 'x0w0x1w1'
# Sum = x0w0x1w1 + b; Sum.label = 'Sum'
# output = Sum.relu(); output.label = 'output'
# output.backward()

# dot = draw_dot(output)
# dot.render("neural_graph", view=True)

x0 = Tensor(1.0, label='x0')
w0 = Tensor(2.0, label='w0')
x1 = Tensor(3.0, label='x1')
w1 = Tensor(4.0, label='w1')
b = Tensor(-7.0, label='b')
x0w0 = x0 * w0; x0w0.label = 'x0w0'
x1w1 = x1 * w1; x1w1.label = 'x1w1'
x0w0x1w1 = x0w0 + x1w1; x0w0x1w1.label = 'x0w0x1w1'
Sum = x0w0x1w1 + b; Sum.label = 'Sum'
output = Sum.tanh(); output.label = 'output'
output.backward()

dot = draw_dot(output)
dot.render("./bp_result/tanh1", view=True)

x0 = Tensor(1.0, label='x0')
w0 = Tensor(2.0, label='w0')
x1 = Tensor(3.0, label='x1')
w1 = Tensor(4.0, label='w1')
b = Tensor(-7.0, label='b')
x0w0 = x0 * w0; x0w0.label = 'x0w0'
x1w1 = x1 * w1; x1w1.label = 'x1w1'
x0w0x1w1 = x0w0 + x1w1; x0w0x1w1.label = 'x0w0x1w1'
Sum = x0w0x1w1 + b; Sum.label = 'Sum'
o = 2 * Sum
exp = o.exp()
output = (exp - 1) / (exp + 1)
output.backward()

dot = draw_dot(output)
dot.render("./bp_result/tanh2", view=True)