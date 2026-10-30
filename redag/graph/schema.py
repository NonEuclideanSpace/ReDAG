import math
from dataclasses import dataclass, field
from graphlib import TopologicalSorter
from typing import Dict, List, Literal, Optional, Set, Tuple
import torch

NodeType = Literal['background', 'entity', 'attribute', 'relation', 'count', 'prohibition']
EdgeType = Literal['comp', 'supp', 'int']
FailureMode = Literal['missing', 'attribute_error', 'relation_error', 'over_count', 'entangled']
OpType = Literal['add_entity', 'modify_attr', 'fix_relation', 'remove_extra', 'separate_instances']
BBox = Tuple[float, float, float, float]


def validate_box(box):
    if len(box) != 4 or not all(math.isfinite(float(x)) for x in box):
        raise ValueError('Box must contain four finite coordinates')
    if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
        raise ValueError('Box must have positive area in original normalized coordinates')


@dataclass
class ConditionNode:
    node_id: str
    node_type: NodeType
    question: str
    entity_phrase: str
    bbox: Optional[BBox] = None
    weight: float = 1.0
    threshold: float = 0.70
    expected_count: Optional[int] = None

    def __post_init__(self):
        if not self.node_id or not self.question.strip():
            raise ValueError('Node ID and question are required')
        if self.node_type not in NodeType.__args__:
            raise ValueError(f'Unknown node type: {self.node_type}')
        if not math.isfinite(self.weight) or self.weight <= 0 or not 0 <= self.threshold <= 1:
            raise ValueError('Invalid node weight or threshold')
        if self.bbox is not None:
            self.bbox = tuple(float(x) for x in self.bbox)
            validate_box(self.bbox)
        if self.expected_count is not None and (type(self.expected_count) is not int or self.expected_count < 0):
            raise ValueError('Expected count must be a nonnegative integer')


@dataclass
class ConditionDAG:
    nodes: Dict[str, ConditionNode]
    comp_edges: List[Tuple[str, str]] = field(default_factory=list)
    supp_edges: List[Tuple[str, str]] = field(default_factory=list)
    int_edges: List[Tuple[str, str]] = field(default_factory=list)

    def __post_init__(self):
        if not self.nodes or any(k != v.node_id for k, v in self.nodes.items()):
            raise ValueError('Graph must contain consistently keyed nodes')
        parents = {k: set() for k in self.nodes}
        for edges in (self.comp_edges, self.supp_edges, self.int_edges):
            for u, v in edges:
                if u not in self.nodes or v not in self.nodes or u == v:
                    raise ValueError('Invalid graph edge')
        for u, v in self.comp_edges + self.supp_edges:
            parents[v].add(u)
        tuple(TopologicalSorter(parents).static_order())

    @property
    def total_weight_Z(self) -> float:
        return float(sum(n.weight for n in self.nodes.values()))


@dataclass
class ControllerAction:
    op: OpType
    box: BBox
    instruction: str

    def __post_init__(self):
        self.box = tuple(float(x) for x in self.box)
        validate_box(self.box)
        if self.op not in OpType.__args__ or not self.instruction.strip():
            raise ValueError('Invalid operation or empty instruction')


@dataclass
class CanvasState:
    image: torch.Tensor
    scores: Dict[str, float]
    protected_set: Set[str]
    step_idx: int = 0

    def __post_init__(self):
        if self.image.ndim != 3 or self.image.shape[0] != 3 or min(self.image.shape[1:]) < 1:
            raise ValueError('Canvas must be an RGB CHW tensor')
        if not self.image.is_floating_point() or not torch.isfinite(self.image).all() or not ((self.image >= 0) & (self.image <= 1)).all():
            raise ValueError('Canvas must contain finite pixels in [0, 1]')
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in self.scores.values()):
            raise ValueError('Scores must be finite probabilities')
        if not self.protected_set <= self.scores.keys() or self.step_idx < 0:
            raise ValueError('Invalid canvas bookkeeping')
