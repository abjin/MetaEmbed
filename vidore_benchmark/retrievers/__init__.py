# Only modules shipped in this repo are imported here. The upstream vidore-benchmark
# retrievers (bge_m3, bm25, cohere, dse, jina, nomic, siglip, ...) are not included.
from .base_vision_retriever import BaseVisionRetriever
from .biqwen2_retriever import BiQwen2Retriever
from .colpali_retriever import ColPaliRetriever
from .colqwen2_retriever import ColQwen2Retriever
from .dummy_vision_retriever import DummyVisionRetriever
from .registry_utils import (
    load_vision_retriever_from_registry,
    register_vision_retriever,
    VISION_RETRIEVER_REGISTRY,
)
from .vision_retriever import DistributedVisionRetriever, VisionRetriever
from .vision_retriever_v2 import (
    DistributedVisionRetriever as DistributedVisionRetrieverV2,
    VisionRetriever as VisionRetrieverV2,
)
