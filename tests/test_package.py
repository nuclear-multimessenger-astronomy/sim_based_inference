from importlib import resources

from nmma.mlmodel import SimilarityEmbedding


def test_public_embedding_api_is_importable():
    assert SimilarityEmbedding.__name__ == "SimilarityEmbedding"


def test_bundled_model_weights_are_package_data():
    package = resources.files("nmma.mlmodel")

    assert package.joinpath("frozen-flow-weights.pth").is_file()
    assert package.joinpath("similarity_embedding_weights.pth").is_file()