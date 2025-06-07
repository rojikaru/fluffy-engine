def local_openclip_ef():
    """
    Create an OpenCLIP embedding function for local use.
    It supports both text and image embeddings.

    :return: OpenCLIPEmbeddingFunction
    """

    from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
    return OpenCLIPEmbeddingFunction()

