import torch
import torch.nn.functional as F
from typing import Dict, Set, Tuple
from redag.graph.schema import BBox, ConditionDAG, validate_box

def defnorm_box_to_pixel_slice(box: BBox, H: int, W: int) -> Tuple[int, int, int, int]:
    validate_box(box)
    if H < 1 or W < 1:
        raise ValueError("Image dimensions must be positive")
    X1 = max(0, min(W - 1, int(round(box[0] * W))))
    Y1 = max(0, min(H - 1, int(round(box[1] * H))))
    X2 = max(X1 + 1, min(W, int(round(box[2] * W))))
    Y2 = max(Y1 + 1, min(H, int(round(box[3] * H))))
    return (X1, Y1, X2, Y2)

def build_inner_feather_mask(h_win: int, w_win: int, feather_px: int=8, device: torch.device=torch.device('cpu')) -> torch.Tensor:
    eff_f = min(feather_px, h_win // 4, w_win // 4)
    if eff_f <= 0:
        return torch.ones((1, h_win, w_win), dtype=torch.float32, device=device)
    y_dist = torch.minimum(torch.arange(h_win, device=device), torch.arange(h_win - 1, -1, -1, device=device)).float()
    x_dist = torch.minimum(torch.arange(w_win, device=device), torch.arange(w_win - 1, -1, -1, device=device)).float()
    y_ramp = torch.clamp(y_dist / float(eff_f), 0.0, 1.0)
    x_ramp = torch.clamp(x_dist / float(eff_f), 0.0, 1.0)
    return (y_ramp[:, None] * x_ramp[None, :]).unsqueeze(0)

def paste_window_strictly_local(canvas: torch.Tensor, edited_crop: torch.Tensor, window_box: BBox, feather_px: int=8, editable: torch.Tensor | None=None) -> torch.Tensor:
    if edited_crop.ndim != 3 or edited_crop.shape[0] != 3 or not torch.isfinite(edited_crop).all():
        raise ValueError("Edited crop must be a finite RGB tensor")
    edited_crop = edited_crop.to(canvas).clamp(0, 1)
    _, H, W = canvas.shape
    X1, Y1, X2, Y2 = defnorm_box_to_pixel_slice(window_box, H, W)
    h_win, w_win = (Y2 - Y1, X2 - X1)
    resized_crop = F.interpolate(edited_crop.unsqueeze(0), size=(h_win, w_win), mode='bilinear', align_corners=False).squeeze(0)
    alpha = build_inner_feather_mask(h_win, w_win, feather_px=feather_px, device=canvas.device)
    if editable is not None:
        if editable.shape[-2:] != (H, W) or not torch.all((editable == 0) | (editable == 1)):
            raise ValueError("The editable mask must be a binary full-canvas mask")
        alpha = alpha * editable.reshape(-1, H, W)[-1:, Y1:Y2, X1:X2].to(alpha)
    out = canvas.clone()
    out[:, Y1:Y2, X1:X2] = alpha * resized_crop + (1.0 - alpha) * canvas[:, Y1:Y2, X1:X2]
    return out

def check_local_acceptance(dag: ConditionDAG, target_node_id: str, prev_scores: Dict[str, float], new_scores: Dict[str, float], protected_set: Set[str], affected_subgraph: Set[str], min_target_gain: float=0.05, max_protected_drop: float=0.08) -> bool:
    tau_u = dag.nodes[target_node_id].threshold
    delta_u = new_scores[target_node_id] - prev_scores[target_node_id]
    target_ok = delta_u >= min_target_gain or new_scores[target_node_id] >= tau_u
    if not target_ok:
        return False
    for v in protected_set & affected_subgraph:
        if new_scores[v] - prev_scores[v] < -max_protected_drop:
            return False
    return True
