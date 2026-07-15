import pytest
import torch

from src.fusion.gated import GatedFusion


def test_output_dim_equals_branch_dim():
    fusion = GatedFusion(
        modality_dims={"endpoint_red": 10, "service_metric": 5, "service_log": 3},
        branch_dim=16,
    )
    assert fusion.output_dim == 16


def test_forward_shape():
    fusion = GatedFusion(
        modality_dims={"endpoint_red": 10, "service_metric": 5, "service_log": 3},
        branch_dim=16,
    )
    batch = {
        "endpoint_red": torch.randn(4, 10),
        "service_metric": torch.randn(4, 5),
        "service_log": torch.randn(4, 3),
    }
    out = fusion(batch)
    assert out.shape == (4, 16)


def test_missing_modality_key_raises():
    with pytest.raises(ValueError):
        GatedFusion(modality_dims={"endpoint_red": 10, "service_metric": 5}, branch_dim=16)


def test_extra_modality_key_raises():
    with pytest.raises(ValueError):
        GatedFusion(
            modality_dims={
                "endpoint_red": 10,
                "service_metric": 5,
                "service_log": 3,
                "extra": 2,
            },
            branch_dim=16,
        )


def test_gate_formula_matches_manual_computation():
    """构造已知权重，手算 g 和 z，验证 forward 输出与公式严格一致。

    公式：e_ep = ReLU(ep_encoder(endpoint_red))
          e_svc = ReLU(svc_encoder(concat(metric, log)))
          g = sigmoid(gate(concat(e_ep, e_svc)))
          z = e_ep + g * value(e_svc)

    输入/权重刻意打破两处对称性，否则本测试无法检测出角色互换类 bug：
      1. e_ep(=0.6) 与 e_svc(=1.2) 数值不同——若二者相等，"z=e_svc+g*value(e_ep)"
         的残差互换 bug 会与正确公式输出一致，测试无法区分。
      2. gate 权重矩阵按 e_ep/e_svc 两半设不同值（0.05 / 0.09）——若整块权重
         用同一个标量 fill，gate(concat(e_ep,e_svc)) 与 gate(concat(e_svc,e_ep))
         在数学上恒等（线性层输出只依赖两半各自的和），拼接顺序互换 bug 也测不出来。
    """
    branch_dim = 2
    fusion = GatedFusion(
        modality_dims={"endpoint_red": 3, "service_metric": 2, "service_log": 1},
        branch_dim=branch_dim,
    )
    fusion.eval()

    # 固定所有可学习权重为已知值，便于手算期望输出。
    with torch.no_grad():
        fusion.ep_encoder[0].weight.fill_(0.1)
        fusion.svc_encoder[0].weight.fill_(0.2)
        fusion.gate.weight[:, :branch_dim].fill_(0.05)
        fusion.gate.weight[:, branch_dim:].fill_(0.09)
        fusion.gate.bias.fill_(0.0)
        fusion.value.weight.fill_(0.3)
        fusion.value.bias.fill_(0.0)

    batch = {
        "endpoint_red": torch.tensor([[1.0, 2.0, 3.0]]),
        "service_metric": torch.tensor([[2.0, 3.0]]),
        "service_log": torch.tensor([[1.0]]),
    }

    with torch.no_grad():
        e_ep = torch.relu(fusion.ep_encoder[0](batch["endpoint_red"]))
        svc_in = torch.cat([batch["service_metric"], batch["service_log"]], dim=-1)
        e_svc = torch.relu(fusion.svc_encoder[0](svc_in))
        g_expected = torch.sigmoid(fusion.gate(torch.cat([e_ep, e_svc], dim=-1)))
        z_expected = e_ep + g_expected * fusion.value(e_svc)

    out = fusion(batch)
    torch.testing.assert_close(out, z_expected)
