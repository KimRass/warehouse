- Reference: https://discuss.pytorch.org/t/how-are-optimizer-step-and-loss-backward-related/7350
```python
for epoch in range(1, n_epochs + 1):
    ...

    # Set each parameter's gradient zero.
    # Calling `loss.backward()` mutiple times accumulates the gradient (by addition) for each parameter. This is why you should call `optimizer.zero_grad()` after each `optimizer.step()` call.
    optimizer.zero_grad()

    # Backpropogate.
    loss.backward()
    # Perform a parameter update based on the current gradient (stored in `.grad` attribute of a parameter)
    optimizer.step()
```