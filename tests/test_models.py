import torch

from nmma.mlmodel.embedding import SimilarityEmbedding, VICRegLoss
from nmma.mlmodel.resnet import ResNet, convN


def test_convN_rejects_even_kernel_sizes():
    try:
        convN(3, 4, kernel_size=2)
    except ValueError as exc:
        assert "even sized kernels" in str(exc)
    else:
        raise AssertionError("convN accepted an even kernel size")


def test_resnet_forward_has_expected_context_shape():
    model = ResNet(num_ifos=(3, None), context_dim=4, layers=[1, 1])

    output = model(torch.randn(2, 3, 32))

    assert output.shape == (2, 4)


def test_similarity_embedding_forward_returns_embedding_and_representation():
    model = SimilarityEmbedding(num_dim=3, num_dim_final=5)

    with torch.no_grad():
        embedding, representation = model(torch.randn(2, 3, 121))

    assert embedding.shape == (2, 5)
    assert representation.shape == (2, 3)


def test_vicreg_loss_returns_scalar_components():
    loss, representation, covariance, standard_deviation = VICRegLoss()(
        torch.randn(4, 5), torch.randn(4, 5)
    )

    for value in (loss, representation, covariance, standard_deviation):
        assert value.ndim == 0
        assert torch.isfinite(value)