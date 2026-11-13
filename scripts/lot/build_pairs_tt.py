#!/usr/bin/env python3
"""Training pairs for the two-track model: HR latent, LR latent, and face / subject masks.

    python3 lot/build_pairs_tt.py --cache DIR --out DIR [--subject yolo] [--items N]

Input: an H3 still cache from ``build_cache_h3.py`` (``*_minimaxh3.safetensors`` latents with
``*_minimaxh3_te.safetensors`` text; ``--cache`` is repeatable). For each item:

- ``hr``: the cached latent, cropped so the LR track has whole 2x2 patches (sides a multiple of 4).
- ``lr``: the same picture at half resolution, as the paper builds its pairs (Sec. H): decode,
  area-downsample x0.5, encode (``VaeUpscaler.down``).
- ``face``: faces found on the decoded HR picture (OpenCV YuNet; ``--detector haar`` needs no
  download), as a mask on the LR token grid (one LR token is 64x64 HR pixels), grown by
  ``--grow`` tokens so eyes and mouth are not at the HR/LR seam.
- ``subject``: with ``--subject yolo``, people found by an Ultralytics segmentation model, on the
  same grid (the paper labels the main subject with an LLM + Grounded SAM; people are the subject
  in this data).

Writes ``<stem>_tt.safetensors`` and ``index.json`` (stem, pair file, text file, caption, face
and subject token counts) under ``--out``. Resumable. The masks are what ``train_tt.py lora``
puts at HR, what ``train_tt.py region`` learns to predict, and what ``train_tt.py decoder`` merges
around.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
TOKEN_PX = 64          # one LR token in HR pixels (HR latent px = 16, LR latent px = 32, token = 2x2)


class FaceDetector:
    def __init__(self, kind: str, score: float = 0.7):
        import cv2

        self.cv2, self.kind = cv2, kind
        if kind == "yunet":
            path = Path.home() / ".cache" / "lot" / "face_detection_yunet_2023mar.onnx"
            if not path.is_file():
                path.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(YUNET_URL, path)
            self.model = cv2.FaceDetectorYN.create(str(path), "", (320, 320), score)
        else:
            self.model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def boxes(self, rgb: np.ndarray) -> list[tuple[float, float, float, float]]:
        """Face boxes ``(x0, y0, x1, y1)`` in pixels."""
        bgr = self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR)
        if self.kind == "yunet":
            self.model.setInputSize((bgr.shape[1], bgr.shape[0]))
            _ok, faces = self.model.detect(bgr)
            found = [] if faces is None else faces[:, :4].tolist()
        else:
            gray = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2GRAY)
            found = self.model.detectMultiScale(gray, 1.1, 5, minSize=(24, 24))
            found = [list(map(float, f)) for f in found]
        return [(x, y, x + w, y + h) for x, y, w, h in found]


class PersonSegmenter:
    def __init__(self, model: str):
        from ultralytics import YOLO

        self.model = YOLO(model)

    def mask(self, rgb: np.ndarray) -> np.ndarray:
        """``(H, W)`` bool: union of person masks."""
        result = self.model.predict(rgb[:, :, ::-1], classes=[0], verbose=False, retina_masks=True)[0]
        if result.masks is None:
            return np.zeros(rgb.shape[:2], dtype=bool)
        return result.masks.data.any(0).cpu().numpy().astype(bool)


# Caption words that name a COCO class (the paper picks the main subject from the prompt with an LLM).
SYNONYMS = {
    "person": ("person", "people", "woman", "women", "man", "men", "girl", "girls", "boy", "boys", "child",
               "children", "kid", "kids", "lady", "guy", "bride", "groom", "couple", "model", "dancer", "chef",
               "worker", "student", "player", "family", "baby", "teenager", "athlete", "musician", "figure",
               "character", "portrait", "supermodel", "farmer", "doctor", "soldier", "knight", "wizard", "teacher",
               "violinist", "singer", "artist", "painter", "scientist", "mother", "father", "grandmother",
               "grandfather", "warrior", "viking", "priest", "nurse", "pilot", "astronaut", "explorer", "hiker",
               "traveler", "tourist", "businessman", "businesswoman", "influencer", "actor", "actress", "mechanic",
               "miner", "blacksmith", "baker", "barista", "gardener", "fisherman", "sailor", "queen", "king",
               "princess", "prince", "monk", "witch", "elf", "dwarf", "hero", "villain", "superhero", "girlfriend",
               "boyfriend", "wife", "husband", "friends", "crowd", "audience", "guest", "guests", "host"),
    "dining table": ("table", "desk"), "potted plant": ("plant", "plants", "flowers", "bonsai"),
    "tv": ("tv", "television", "screen", "monitor"), "couch": ("couch", "sofa"), "motorcycle": ("motorbike",),
    "airplane": ("plane", "aircraft"), "cell phone": ("phone", "smartphone"),
}


class SubjectSegmenter:
    """Main subject: YOLO instance masks of any of its 80 classes, preferring classes the caption names,
    ranked by confidence, size and centrality; same-class instances close to the best are kept too."""

    def __init__(self, model: str, conf: float = 0.35, center: float = 0.8):
        from ultralytics import YOLO

        self.model, self.conf, self.center = YOLO(model), conf, center
        self.names = self.model.names

    def words(self, caption: str) -> set[str]:
        import re

        tokens = set(re.findall(r"[a-z]+", caption.lower()))
        named = set()
        for index, name in self.names.items():
            options = SYNONYMS.get(name, ()) + (name, name + "s")
            if any(" " not in o and o in tokens for o in options) or (" " in name and name in caption.lower()):
                named.add(name)
        return named

    def mask(self, rgb: np.ndarray, caption: str = "") -> tuple[np.ndarray, str]:
        height, width = rgb.shape[:2]
        result = self.model.predict(rgb[:, :, ::-1], conf=self.conf, verbose=False, retina_masks=True)[0]
        if result.masks is None or len(result.boxes) == 0:
            return np.zeros((height, width), dtype=bool), ""
        masks = result.masks.data.cpu().numpy().astype(bool)
        classes = [self.names[int(c)] for c in result.boxes.cls.tolist()]
        confs = result.boxes.conf.tolist()
        named = self.words(caption)
        pool = [i for i, c in enumerate(classes) if c in named] or list(range(len(classes)))
        yy, xx = np.mgrid[0:height, 0:width]

        def score(i: int) -> float:
            area = masks[i].mean()
            if area == 0:
                return 0.0
            cy, cx = yy[masks[i]].mean() / height - 0.5, xx[masks[i]].mean() / width - 0.5
            return confs[i] * area ** 0.5 * (1.0 - self.center * (cy * cy + cx * cx) ** 0.5)

        best = max(pool, key=score)
        keep = [i for i in pool if classes[i] == classes[best] and score(i) >= 0.5 * score(best)]
        return np.any(masks[keep], axis=0), classes[best]


STOPWORDS = set("""a an the and or of in on at to for with without from by into onto over under near behind beside
between among around through across along up down out off is are was were be been being this that these those it its
their his her them they he she we you your our my as while than then there here very more most some any each every all
both few many much other another such own same so too just only also not no nor can will would should could may might
featuring featured features showing shows shown displaying depicting set sits sitting standing stands lying posing
poses looking holding wearing walking dressed made using surrounded filled covered decorated placed seen view views
scene scenes style styled image photo photograph picture shot render rendering illustration artwork design concept
isometric photorealistic realistic detailed highly ultra high quality 3d 2d cozy luxurious luxury modern vintage
classic rustic elegant stylish beautiful stunning gorgeous captivating charming vibrant vivid colorful soft warm cool
cold bright dark dim moody dramatic cinematic natural ambient golden pastel lighting light lights lit glow glowing
background foreground backdrop setting atmosphere mood tone tones color colors colour black white small large big tiny
huge little tall short long wide young old new front side top bottom left right center middle inside outside interior
exterior close closeup close-up full half wall floor ceiling day night time morning evening sunset sunrise room
rooms living area space diorama miniature technical illustration render cutaway cross-section section interior-design
landscape cityscape environment world place location surroundings ambience vibe aesthetic theme themed""".split())


class YoloeSubject:
    """Main subject with YOLOE (open-vocabulary segmentation), prompted with the caption's content words.

    Closer to the paper's LLM + Grounded SAM than a fixed 80-class detector: "a lighthouse on a cliff" can
    segment the lighthouse. Words earlier in the caption rank higher (the subject usually comes first).
    Falls back to ``fallback`` (the COCO ranking) when no prompt finds anything."""

    def __init__(self, model: str = "yoloe-11s-seg.pt", conf: float = 0.25, max_words: int = 8, fallback=None):
        from ultralytics import YOLOE

        self.model, self.conf, self.max_words, self.fallback = YOLOE(model), conf, max_words, fallback
        self._names: list[str] | None = None

    def candidates(self, caption: str) -> list[str]:
        import re

        words, out = re.findall(r"[a-z][a-z-]+", caption.lower()), []
        for word in words:
            if len(word) < 3 or word in STOPWORDS:
                continue
            word = "person" if word in SYNONYMS["person"] else word
            if word not in out:
                out.append(word)
        return out[:self.max_words] or ["person"]

    def mask(self, rgb: np.ndarray, caption: str = "") -> tuple[np.ndarray, str]:
        """Every person in the frame, plus the best instance of a non-person thing the caption names.

        People always count (where viewers look); position does not (no centrality term), so the
        head cannot learn "the subject is in the middle" from the labels."""
        height, width = rgb.shape[:2]
        named = self.candidates(caption)
        names = ["person"] + [n for n in named if n != "person"]
        result, masks, classes, confs = self._detect(rgb, names)
        if not classes:
            return (self.fallback.mask(rgb, caption) if self.fallback is not None
                    else (np.zeros((height, width), dtype=bool), ""))
        people = [i for i, c in enumerate(classes) if c == "person" and masks[i].any()]
        things = [i for i, c in enumerate(classes) if c != "person" and 0 < masks[i].mean() <= 0.35]

        def score(i: int) -> float:
            return confs[i] * masks[i].mean() ** 0.5 * (1.0 - 0.08 * names.index(classes[i]))

        keep = list(people)
        label = "person" if people else ""
        if things:
            best = max(things, key=score)
            keep.append(best)
            label = f"{label}+{classes[best]}" if label else classes[best]
        if not keep:
            return (self.fallback.mask(rgb, caption) if self.fallback is not None
                    else (np.zeros((height, width), dtype=bool), ""))
        return np.any(masks[keep], axis=0), label

    def _detect(self, rgb: np.ndarray, names: list[str]):
        if names != self._names:
            self.model.set_classes(names, self.model.get_text_pe(names))
            self._names = names
        result = self.model.predict(rgb[:, :, ::-1], conf=self.conf, verbose=False, retina_masks=True)[0]
        if result.masks is None or len(result.boxes) == 0:
            return result, None, [], []
        masks = result.masks.data.cpu().numpy().astype(bool)
        return result, masks, [names[int(c)] for c in result.boxes.cls.tolist()], result.boxes.conf.tolist()

    def mask_v1(self, rgb: np.ndarray, caption: str = "") -> tuple[np.ndarray, str]:
        """The first rule (one best subject, centrality-weighted); kept for comparison."""
        height, width = rgb.shape[:2]
        names = self.candidates(caption)
        if names != self._names:
            self.model.set_classes(names, self.model.get_text_pe(names))
            self._names = names
        result = self.model.predict(rgb[:, :, ::-1], conf=self.conf, verbose=False, retina_masks=True)[0]
        if result.masks is None or len(result.boxes) == 0:
            if self.fallback is not None:
                return self.fallback.mask(rgb, caption)
            return np.zeros((height, width), dtype=bool), ""
        masks = result.masks.data.cpu().numpy().astype(bool)
        classes = [names[int(c)] for c in result.boxes.cls.tolist()]
        confs = result.boxes.conf.tolist()
        yy, xx = np.mgrid[0:height, 0:width]

        def score(i: int) -> float:
            area = masks[i].mean()
            if area == 0 or area > 0.35:                      # scene-sized masks ("hospital", "underwater") are not subjects
                return 0.0
            cy, cx = yy[masks[i]].mean() / height - 0.5, xx[masks[i]].mean() / width - 0.5
            order = 1.0 - 0.08 * names.index(classes[i])        # earlier caption words first
            people = 1.5 if classes[i] == "person" else 1.0       # where viewers look first
            return confs[i] * area ** 0.5 * (1.0 - 0.8 * (cy * cy + cx * cx) ** 0.5) * order * people

        best = max(range(len(classes)), key=score)
        if score(best) <= 0.0:
            return (self.fallback.mask(rgb, caption) if self.fallback is not None
                    else (np.zeros((height, width), dtype=bool), ""))
        keep = [i for i in range(len(classes)) if classes[i] == classes[best] and score(i) >= 0.5 * score(best)]
        return np.any(masks[keep], axis=0), classes[best]


def caption_for(te_path: str) -> str:
    """The caption of a cache item, from the cache's captions.json (empty if missing)."""
    te = Path(te_path)
    path = te.parent / "captions.json"
    if not path.is_file():
        return ""
    if not hasattr(caption_for, "cache") or caption_for.cache[0] != path:
        caption_for.cache = (path, json.loads(path.read_text()))
    return caption_for.cache[1].get(te.name.replace("_minimaxh3_te.safetensors", ""), "")


def boxes_to_tokens(boxes, height: int, width: int, grow: int) -> torch.Tensor:
    """Pixel boxes -> ``(1, h, w)`` bool on the LR token grid, grown by ``grow`` tokens."""
    tok_h, tok_w = height // TOKEN_PX, width // TOKEN_PX
    mask = torch.zeros(1, tok_h, tok_w, dtype=torch.bool)
    for x0, y0, x1, y1 in boxes:
        u0, v0 = max(0, int(y0 // TOKEN_PX)), max(0, int(x0 // TOKEN_PX))
        u1, v1 = min(tok_h, int(np.ceil(y1 / TOKEN_PX))), min(tok_w, int(np.ceil(x1 / TOKEN_PX)))
        mask[0, u0:u1, v0:v1] = True
    return grow_tokens(mask, grow)


def pixels_to_tokens(pixel_mask: np.ndarray, grow: int, share: float = 0.25) -> torch.Tensor:
    """``(H, W)`` bool -> ``(1, h, w)``: tokens at least ``share`` covered, grown by ``grow``."""
    field = torch.from_numpy(pixel_mask).float()[None, None]
    tok_h, tok_w = pixel_mask.shape[0] // TOKEN_PX, pixel_mask.shape[1] // TOKEN_PX
    cover = F.adaptive_avg_pool2d(field[..., :tok_h * TOKEN_PX, :tok_w * TOKEN_PX], (tok_h, tok_w))[0]
    return grow_tokens(cover >= share, grow)


def grow_tokens(mask: torch.Tensor, grow: int) -> torch.Tensor:
    if grow <= 0 or not mask.any():
        return mask
    return F.max_pool2d(mask.float()[:, None], 2 * grow + 1, stride=1, padding=grow)[:, 0] > 0.5


def list_pairs(pairs_dirs, holdout: int = 8) -> tuple[list[dict], list[dict]]:
    """``index.json`` entries of one or more pair dirs, split deterministically into train / held out."""
    entries = []
    for root in pairs_dirs:
        root = Path(root)
        for entry in json.loads((root / "index.json").read_text()):
            entries.append({**entry, "pair": str(root / entry["pair"])})
    if len(entries) <= holdout:
        raise SystemExit(f"{len(entries)} pairs, need more than --holdout {holdout}")
    random.Random(0).shuffle(entries)
    return entries[holdout:], entries[:holdout]


def load_pair(entry: dict, device="cuda") -> dict:
    """``hr`` ``(1, 24, 1, H, W)``, ``lr`` ``(1, 24, 1, H/2, W/2)``, ``face`` / ``subject`` ``(1, h, w)`` bool, ``text``."""
    from safetensors import safe_open

    out = {}
    with safe_open(entry["pair"], "pt") as handle:
        for key in handle.keys():
            out[key] = handle.get_tensor(key)
    with safe_open(entry["te"], "pt") as handle:
        hidden, mask = handle.get_tensor("hidden_states"), handle.get_tensor("attention_mask")
    return {"hr": out["hr"].float()[None].to(device), "lr": out["lr"].float()[None].to(device),
            "face": out["face"].bool(), "subject": out["subject"].bool(),
            "text": hidden[mask].unsqueeze(0).to(device, torch.bfloat16)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--items", type=int, default=0, help="0 = all")
    parser.add_argument("--detector", choices=("yunet", "haar"), default="yunet")
    parser.add_argument("--subject", choices=("none", "yolo", "main", "yoloe"), default="none",
                        help="yolo: people only; main: the main subject, any of YOLO's 80 classes, caption-guided; "
                             "yoloe: open-vocabulary YOLOE prompted with the caption's words (falls back to main)")
    parser.add_argument("--yoloe-model", default="yoloe-11s-seg.pt")
    parser.add_argument("--photos", type=Path, action="append", default=None,
                        help="relabel: folders of the source photos (cover-resized and centre-cropped like "
                             "build_cache_h3.py media), used instead of decoding each latent")
    parser.add_argument("--tag", default=None, help="relabel: tag written to index entries (resume key)")
    parser.add_argument("--relabel", action="store_true",
                        help="recompute face/subject masks of existing pairs in --out (no re-encoding)")
    parser.add_argument("--yolo-model", default="yolo11n-seg.pt")
    parser.add_argument("--grow", type=int, default=1, help="tokens added around each face")
    parser.add_argument("--subject-grow", type=int, default=0)
    args = parser.parse_args()

    from gpu_guard import refuse_if_busy
    from two_track_h3 import load_upscaler

    refuse_if_busy("build_pairs_tt.py")
    if args.relabel:
        relabel(args)
        return
    items = []
    for cache in args.cache:
        items += _all_items(cache)
    if args.items:
        items = items[:args.items]
    args.out.mkdir(parents=True, exist_ok=True)
    index_path = args.out / "index.json"
    index = {e["stem"]: e for e in json.loads(index_path.read_text())} if index_path.is_file() else {}
    vae = load_upscaler()
    faces = FaceDetector(args.detector)
    people = PersonSegmenter(args.yolo_model) if args.subject == "yolo" else None
    subjects = make_subjects(args)
    from safetensors import safe_open
    from safetensors.torch import save_file

    for number, (latent_path, te_path) in enumerate(items):
        stem = latent_path.name.replace("_minimaxh3.safetensors", "")
        if stem in index and (args.out / index[stem]["pair"]).is_file():
            continue
        with safe_open(str(latent_path), "pt") as handle:
            latent = handle.get_tensor(next(k for k in handle.keys() if k.startswith("latent"))).float()
        if latent.ndim == 3:
            latent = latent.unsqueeze(1)
        if latent.shape[1] != 1:
            print(f"LOT_H3 kind=pairs_skip stem={stem} reason=clip (stills only)", flush=True)
            continue
        height, width = latent.shape[-2] // 4 * 4, latent.shape[-1] // 4 * 4
        hr = latent[..., :height, :width].unsqueeze(0).cuda()
        with torch.no_grad():
            lr = vae.down(hr)
            pixels = vae.vae.decode(hr.to(vae.dtype))[0, :, 0].float().clamp(-1, 1)
        rgb = ((pixels.permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).round().astype(np.uint8)
        face = boxes_to_tokens(faces.boxes(rgb), rgb.shape[0], rgb.shape[1], args.grow)
        subject_class = "person" if people is not None else ""
        if subjects is not None:
            pixel_mask, subject_class = subjects.mask(rgb, caption_for(str(te_path)))
            subject = pixels_to_tokens(pixel_mask, args.subject_grow)
        elif people is not None:
            subject = pixels_to_tokens(people.mask(rgb), args.subject_grow)
        else:
            subject = torch.zeros_like(face)
        pair_name = f"{stem}_tt.safetensors"
        save_file({"hr": hr[0].half().cpu().contiguous(), "lr": lr[0].half().cpu().contiguous(),
                   "face": face.to(torch.uint8), "subject": subject.to(torch.uint8)}, str(args.out / pair_name))
        index[stem] = {"stem": stem, "pair": pair_name, "te": str(te_path), "face_tokens": int(face.sum()),
                       "subject_tokens": int(subject.sum()), "subject_class": subject_class,
                       "tokens": int(face.numel())}
        if number % 25 == 0:
            index_path.write_text(json.dumps(list(index.values()), indent=0))
            print(f"LOT_H3 kind=pairs item={number + 1}/{len(items)} stem={stem} face={int(face.sum())} "
                  f"subject={int(subject.sum())} of {face.numel()}", flush=True)
    index_path.write_text(json.dumps(list(index.values()), indent=0))
    with_face = sum(1 for e in index.values() if e["face_tokens"])
    print(f"LOT_H3 kind=pairs_done out={args.out} pairs={len(index)} with_face={with_face} ok=1", flush=True)


def make_subjects(args):
    """The main-subject segmenter for ``--subject`` (``None`` for none / yolo)."""
    if args.subject == "main":
        return SubjectSegmenter(args.yolo_model)
    if args.subject == "yoloe":
        return YoloeSubject(args.yoloe_model, fallback=SubjectSegmenter(args.yolo_model, center=0.0))
    return None


class _LazyVae:
    """Loads the VAE only if some pair has no source photo."""

    def __getattr__(self, name):
        from two_track_h3 import load_upscaler

        real = load_upscaler()
        self.__dict__.update(real.__dict__)
        self.__class__ = type(real)
        return getattr(real, name)


def photo_for(stem: str, dirs: list[Path], width: int, height: int):
    """The source photo of a cache stem, cover-resized and centre-cropped to ``width x height`` the way
    ``build_cache_h3.py media`` built the latent, as uint8 RGB; ``None`` if not found."""
    import re

    from PIL import Image

    base = re.sub(r"_\d{4}x\d{4}$", "", stem)
    for folder in dirs:
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            path = folder / f"{base}{ext}"
            if path.is_file():
                image = Image.open(path).convert("RGB")
                scale = max(width / image.width, height / image.height)
                size = (max(width, round(image.width * scale)), max(height, round(image.height * scale)))
                image = image.resize(size, Image.BICUBIC)
                left, top = (size[0] - width) // 2, (size[1] - height) // 2
                return np.asarray(image.crop((left, top, left + width, top + height)))
    return None


def relabel(args) -> None:
    """Recompute the masks of existing pairs: decode ``hr``, detect faces and the main subject, rewrite."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    from two_track_h3 import load_upscaler

    index_path = args.out / "index.json"
    entries = json.loads(index_path.read_text())
    vae = load_upscaler() if not args.photos else _LazyVae()
    faces = FaceDetector(args.detector)
    subjects = make_subjects(args) or SubjectSegmenter(args.yolo_model)
    tag = args.tag or ("yoloe" if args.subject == "yoloe" else "main")
    photo_dirs = args.photos or []
    classes: dict[str, int] = {}
    for number, entry in enumerate(entries):
        if entry.get("relabel") == tag and not args.items:
            continue                                               # resumable
        path = args.out / entry["pair"]
        with safe_open(str(path), "pt") as handle:
            tensors = {k: handle.get_tensor(k) for k in handle.keys()}
        height, width = tensors["hr"].shape[-2] * 16, tensors["hr"].shape[-1] * 16
        rgb = photo_for(entry["stem"], photo_dirs, width, height)
        if rgb is None:
            with torch.no_grad():
                pixels = vae.vae.decode(tensors["hr"][None].cuda().to(vae.dtype))[0, :, 0].float().clamp(-1, 1)
            rgb = ((pixels.permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).round().astype(np.uint8)
        face = boxes_to_tokens(faces.boxes(rgb), rgb.shape[0], rgb.shape[1], args.grow)
        pixel_mask, name = subjects.mask(rgb, caption_for(entry["te"]))
        subject = pixels_to_tokens(pixel_mask, args.subject_grow)
        tensors["face"], tensors["subject"] = face.to(torch.uint8), subject.to(torch.uint8)
        save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
        entry.update({"face_tokens": int(face.sum()), "subject_tokens": int(subject.sum()), "subject_class": name,
                      "relabel": tag})
        classes[name or "-"] = classes.get(name or "-", 0) + 1
        if number % 50 == 0:
            index_path.write_text(json.dumps(entries, indent=0))
            print(f"LOT_H3 kind=relabel item={number + 1}/{len(entries)} class={name or '-'} "
                  f"face={int(face.sum())} subject={int(subject.sum())}", flush=True)
        if args.items and number + 1 >= args.items:
            break
    index_path.write_text(json.dumps(entries, indent=0))
    top = sorted(classes.items(), key=lambda kv: -kv[1])[:12]
    print(f"LOT_H3 kind=relabel_done out={args.out} classes={top} ok=1", flush=True)


def _all_items(cache: Path):
    """Every ``(latent, te)`` pair of a cache (the train/held split happens in ``list_pairs``)."""
    import re

    items = []
    for latent in sorted(cache.glob("*_minimaxh3.safetensors")):
        stem = re.sub(r"_\d{4}x\d{4}_minimaxh3\.safetensors$", "", latent.name)
        te = cache / f"{stem}_minimaxh3_te.safetensors"
        if te.is_file():
            items.append((latent, te))
    if not items:
        raise SystemExit(f"{cache}: no latents with _te text (run build_cache_h3.py text)")
    return items


if __name__ == "__main__":
    main()
