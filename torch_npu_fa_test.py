import torch
import torch_npu

torch.npu.set_device(0)

# flash_attn_func 使用 BSND：[batch, seq, heads, head_dim]
q = torch.randn(1, 128, 8, 64, device="npu",
                dtype=torch.float16, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)

# sparse_mode=3 要求 2048×2048 压缩 causal mask
causal_mask = torch.triu(
    torch.ones(2048, 2048, device="npu"), diagonal=1
).bool()

out = torch_npu.npu_fusion_attention(
    q,
    k,
    v,
    q.shape[2],
    "BSND",
    atten_mask=causal_mask,
    scale=q.shape[-1] ** -0.5,
    keep_prob=1.0,
    sparse_mode=3,
)[0]

out.float().sum().backward()
print(out.shape, q.grad is not None)