import torch
import torch.nn as nn
from mep.optimizers import SMEPOptimizer, EPWrapper

def test_new_api_updates_weights():
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2)
    )

    initial_weight = model[0].weight.detach().clone()

    optimizer = SMEPOptimizer(model.parameters(), model=model, lr=0.01, mode='ep', ns_steps=4)

    x = torch.randn(5, 10)
    y = torch.randint(0, 2, (5,))

    optimizer.zero_grad()
    model(x)
    optimizer.step(target=y)

    assert not torch.allclose(model[0].weight, initial_weight), "Weights should have updated"

def test_new_api_backprop_fallback():
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2)
    )

    optimizer = SMEPOptimizer(model.parameters(), model=model, lr=0.01, mode='backprop')

    x = torch.randn(5, 10)
    y = torch.randint(0, 2, (5,))
    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    # Implicit assertion: no error

def test_reinitialization_idempotency():
    model = nn.Sequential(nn.Linear(10, 2))

    # 1. Initialize EP
    opt1 = SMEPOptimizer(model.parameters(), model=model, mode='ep')
    assert hasattr(model.forward, '__self__')
    assert isinstance(model.forward.__self__, EPWrapper)

    # 2. Re-initialize EP
    opt2 = SMEPOptimizer(model.parameters(), model=model, mode='ep')
    assert hasattr(model.forward, '__self__')
    assert isinstance(model.forward.__self__, EPWrapper)
    # Check it's the NEW wrapper
    assert model.forward.__self__.optimizer is opt2

    # 3. Initialize Backprop
    opt3 = SMEPOptimizer(model.parameters(), model=model, mode='backprop')
    # Should be unwrapped
    # Note: hasattr(..., '__self__') might be true for bound methods too, but isinstance check confirms wrapper
    is_wrapper = hasattr(model.forward, '__self__') and isinstance(model.forward.__self__, EPWrapper)
    assert not is_wrapper

    # Verify forward still works
    out = model(torch.randn(1, 10))
    assert out.shape == (1, 2)
