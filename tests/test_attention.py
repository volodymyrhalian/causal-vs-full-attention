import torch
from attention_lab.data import make_dataset
from attention_lab.model import attention_visibility, ModelConfig, ContextTransformer


def test_mask_semantics():
    causal = attention_visibility(7, True)
    full = attention_visibility(7, False)
    assert torch.equal(causal, torch.ones(7, 7, dtype=torch.bool).tril())
    assert full.all()
    assert not causal[2, 3] and causal[3, 2]


def test_parameter_matching():
    cfg = ModelConfig(d_model=32, num_heads=4, num_layers=2)
    a = ContextTransformer(cfg, "causal")
    b = ContextTransformer(cfg, "full")
    assert a.parameter_count() == b.parameter_count()


def test_causal_future_invariance():
    torch.manual_seed(13)
    model = ContextTransformer(ModelConfig(d_model=32, num_heads=4, num_layers=2), "causal").eval()
    data = make_dataset(16, "right", seed=12)
    mutated = data.tokens.clone()
    for i, q in enumerate(data.query_positions):
        mutated[i, q.item() + 1:] = 10
    with torch.no_grad():
        before = model(data.tokens, data.query_positions)
        after = model(mutated, data.query_positions)
    torch.testing.assert_close(before, after, rtol=0, atol=1e-6)


def test_future_weights_are_zero():
    data = make_dataset(3, "right", seed=3)
    model = ContextTransformer(ModelConfig(d_model=32, num_heads=4), "causal").eval()
    with torch.no_grad():
        _, maps = model(data.tokens, data.query_positions, inspect=True)
    for w in maps:
        for i, q in enumerate(data.query_positions):
            assert (w[i, :, q.item(), q.item()+1:] == 0).all()


def test_same_initialized_weights_different_visibility():
    cfg = ModelConfig(d_model=32, num_heads=4, num_layers=2)
    data = make_dataset(4, "right", seed=2)
    torch.manual_seed(0)
    causal = ContextTransformer(cfg, "causal").eval()
    full = ContextTransformer(cfg, "full").eval()
    full.load_state_dict(causal.state_dict())
    with torch.no_grad():
        a = causal(data.tokens, data.query_positions)
        b = full(data.tokens, data.query_positions)
    assert not torch.allclose(a, b)
