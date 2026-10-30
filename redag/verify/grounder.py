from copy import deepcopy
import numpy as np
import torch
from PIL import Image
from redag.graph.dag import compute_box_iou, topological_nodes


def load_image(path):
    with Image.open(path) as image:
        return torch.from_numpy(np.asarray(image.convert('RGB')).copy()).permute(2, 0, 1).float() / 255


def save_image(image, path):
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(path)


class Grounder:
    def __init__(self, detector, segmenter=None):
        self.detector = detector
        self.segmenter = segmenter
        self.counts = {}
        self.masks = {}

    @classmethod
    def from_pretrained(cls, detector_id='IDEA-Research/grounding-dino-base', sam_id='facebook/sam2.1-hiera-large', device='cuda'):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor
        processor = AutoProcessor.from_pretrained(detector_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(detector_id).to(device).eval()
        sam_processor = Sam2Processor.from_pretrained(sam_id)
        sam = Sam2Model.from_pretrained(sam_id).to(device).eval()
        model.requires_grad_(False)
        sam.requires_grad_(False)

        @torch.no_grad()
        def detect(image, phrase):
            _, height, width = image.shape
            rgb = Image.fromarray((image.cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
            inputs = processor(images=rgb, text=phrase.rstrip('.') + '.', do_pad=False, return_tensors='pt').to(device)
            outputs = model(**inputs)
            result = processor.post_process_grounded_object_detection(outputs, inputs.input_ids, threshold=0.30, text_threshold=0.25, target_sizes=[(height, width)])[0]
            order = result['scores'].argsort(descending=True)
            return result['boxes'][order].cpu().tolist()

        @torch.no_grad()
        def segment(image, boxes):
            height, width = image.shape[-2:]
            side = min(height, width)
            if side < 1:
                raise ValueError('Segmentation requires a nonempty image')
            long_side = max(height, width)
            origins = list(range(0, long_side - side + 1, side))
            if origins[-1] != long_side - side:
                origins.append(long_side - side)
            merged = torch.zeros((len(boxes), height, width), dtype=torch.bool)
            for origin in origins:
                left, top = (origin, 0) if width >= height else (0, origin)
                local_boxes, indices = [], []
                for index, box in enumerate(boxes):
                    x1, y1 = max(float(box[0]), left), max(float(box[1]), top)
                    x2, y2 = min(float(box[2]), left + side), min(float(box[3]), top + side)
                    if x2 > x1 and y2 > y1:
                        local_boxes.append([x1 - left, y1 - top, x2 - left, y2 - top])
                        indices.append(index)
                if not indices:
                    continue
                tile = image[:, top:top + side, left:left + side]
                rgb = Image.fromarray((tile.cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
                inputs = sam_processor(images=rgb, input_boxes=[local_boxes], return_tensors='pt').to(device)
                outputs = sam(**inputs, multimask_output=False)
                masks = sam_processor.post_process_masks(outputs.pred_masks.cpu(), inputs['original_sizes'].cpu())[0]
                masks = masks.reshape(len(indices), side, side).bool()
                for index, mask in zip(indices, masks):
                    merged[index, top:top + side, left:left + side] |= mask
            return merged
        return cls(detect, segment)

    def ground(self, image, dag, node_ids=None):
        result = deepcopy(dag)
        selected = set(dag.nodes) if node_ids is None else set(node_ids)
        self.masks = {k: v for k, v in self.masks.items() if k not in selected}
        height, width = image.shape[-2:]
        detections, used = {}, {}
        for nid in topological_nodes(result):
            node = result.nodes[nid]
            if nid not in selected or node.node_type not in ('entity', 'count'):
                continue
            phrase = node.entity_phrase
            if phrase not in detections:
                raw = self.detector(image, phrase)
                boxes = []
                for box in raw:
                    b = tuple(float(max(0, min(1, x / size))) for x, size in zip(box, (width, height, width, height)))
                    if b[0] < b[2] and b[1] < b[3] and not any(compute_box_iou(b, other) > 0.8 for other in boxes):
                        boxes.append(b)
                detections[phrase] = boxes
                self.counts[phrase] = len(boxes)
                used[phrase] = set()
            boxes = detections[phrase]
            if node.node_type == 'count':
                node.bbox = None
                continue
            available = [i for i in range(len(boxes)) if i not in used[phrase]]
            if not available:
                node.bbox = None
                continue
            index = max(available, key=lambda i: compute_box_iou(boxes[i], node.bbox)) if node.bbox else available[0]
            used[phrase].add(index)
            node.bbox = boxes[index]
            if self.segmenter is not None:
                pixel_box = [v * size for v, size in zip(node.bbox, (width, height, width, height))]
                mask = self.segmenter(image, [pixel_box])[0]
                if mask.shape != (height, width):
                    raise ValueError('Segmenter must return original-resolution masks')
                self.masks[nid] = mask
        for nid in topological_nodes(result):
            node = result.nodes[nid]
            if nid not in selected or node.node_type not in ('attribute', 'relation'):
                continue
            parents = [result.nodes[p].bbox for p, c in result.comp_edges if c == nid and result.nodes[p].bbox is not None]
            if node.node_type == 'attribute':
                node.bbox = parents[0] if parents else None
            elif parents:
                node.bbox = (min(b[0] for b in parents), min(b[1] for b in parents), max(b[2] for b in parents), max(b[3] for b in parents))
            else:
                node.bbox = None
        return result
