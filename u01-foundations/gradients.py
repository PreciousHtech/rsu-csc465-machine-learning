"""Unit 1: derivatives, partial derivatives and the gradient.

We compare exact derivatives with numerical ones (the change in f
for a very small change in the input), then take a few steps
downhill on a function of two variables.
"""
import numpy as np


def f(w):                       # one variable: minimum at w = 3
    return (w - 3) ** 2


def g(v):                       # two variables: minimum at (1, -2)
    x, y = v
    return (x - 1) ** 2 + 2 * (y + 2) ** 2


def grad_g(v):                  # exact partial derivatives
    x, y = v
    return np.array([2 * (x - 1), 4 * (y + 2)])


h = 1e-6
print("f'(5) exact = 4, numerical =", round((f(5 + h) - f(5)) / h, 4))

for learningRate in [0.05, 0.4, 0.6]:
 v = np.array([4.0, 1.0])
 print(f"\nlearning rate {learningRate}")    
num = [(g(v + h * e) - g(v)) / h for e in np.eye(2)]
print("gradient of g at (4, 1): exact", grad_g(v),
      "numerical", np.round(num, 3))

for step in range(1, 6):        # gradient descent, learning rate 0.2
    v = v - learningRate * grad_g(v)
    print(f"step {step}: v = {np.round(v, 3)}  g(v) = {g(v):.4f}")

# The y-coordinate flips sign and grows every step (−6.2, 3.88, −10.232, 9.525, −18.135), and g rises from 35.6 to 520.7. which means 0.6 diverges. 
# The x factor is −0.2, so x converges (0.976, 1.005, 0.999) while y diverges in the output.
# The Hessian of g is diagonal: ∂²g/∂x² = 2, ∂²g/∂y² = 4, ∂²g/∂x∂y = 0. So each coordinate's distance from the minimum (1, −2) is multiplied by (1 − α·curvature) every step:
#x: factor 1 − 2α, converges if α < 1
#y: factor 1 − 4α, converges if α < 0.5