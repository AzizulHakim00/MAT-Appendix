from __future__ import annotations

import hashlib
import os
import urllib.request

REPOSITORY = "AzizulHakim00/MAT-Appendix"
PART_NAMES = [
    "cgr_mat_multimodal_pipeline.py.part01",
    "cgr_mat_multimodal_pipeline.py.part02",
    "cgr_mat_multimodal_pipeline.py.part03",
    "cgr_mat_multimodal_pipeline.py.part04",
    "cgr_mat_multimodal_pipeline.py.part05",
    "cgr_mat_multimodal_pipeline.py.part06",
]
PART_SHA256 = {
    "cgr_mat_multimodal_pipeline.py.part01": "7ecd6b0d816a9549d23def5f67e54286ef5865e4bc6a0b74bce020d9a2d2a713",
    "cgr_mat_multimodal_pipeline.py.part02": "042a022fe7652ac4fab86b50ed1aae301d88865c92464188a027d044a32b07b7",
    "cgr_mat_multimodal_pipeline.py.part03": "658a91d1713a304d29e2af914e6b4a8ca4549cf06bf887e6c20d84ca5a3ba41f",
    "cgr_mat_multimodal_pipeline.py.part04": "149f9d6b85bceb743c3dc2f00d68bc1971a683daf4ee9d275115c0e9fade1170",
    "cgr_mat_multimodal_pipeline.py.part05": "4f7cc2cf2f1fda7f830d9c3a3ee7d620f2d3f81abccb76346fef038eea4af0be",
    "cgr_mat_multimodal_pipeline.py.part06": "4cd732ef9067805c43aac9f44650317c8dee2fc19f8ce3945cb149a243860d5e",
}
ORIGINAL_COMBINED_SHA256 = "bf7a8b0c4fbb6020411bb73ae1769d2e89ea068685ac3a905546df904e32b864"
PATCHED_COMBINED_SHA256 = "fcbc79b8b716e42a1f089433826990d8128f6655cfc182a280e50e40b4dd558d"

SOURCE_COMMIT = globals().get("SOURCE_COMMIT") or os.environ.get("CGR_MAT_SOURCE_COMMIT")
if not SOURCE_COMMIT:
    raise RuntimeError("SOURCE_COMMIT must be pinned by the notebook before loading the pipeline.")

print("Loading checksum-pinned CGR-MAT source parts...")
print("Source commit:", SOURCE_COMMIT)
chunks: list[str] = []
for index, name in enumerate(PART_NAMES, start=1):
    url = (
        f"https://raw.githubusercontent.com/{REPOSITORY}/{SOURCE_COMMIT}/"
        f"src/cgr_mat/parts/{name}"
    )
    raw = urllib.request.urlopen(url, timeout=120).read()
    actual = hashlib.sha256(raw).hexdigest()
    expected = PART_SHA256[name]
    if actual != expected:
        raise RuntimeError(
            f"Source part integrity failure for {name}.\n"
            f"Expected: {expected}\nActual:   {actual}"
        )
    chunks.append(raw.decode("utf-8"))
    print(f"  ✓ part {index}/{len(PART_NAMES)} verified: {name}")

source = "".join(chunks)
original_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
if original_sha != ORIGINAL_COMBINED_SHA256:
    raise RuntimeError(
        "Original combined source integrity failure.\n"
        f"Expected: {ORIGINAL_COMBINED_SHA256}\nActual:   {original_sha}"
    )

# Deterministic safety patches are applied only after the original source has
# passed all part-level and combined integrity checks. Each target must occur
# exactly once; otherwise execution stops rather than applying an ambiguous edit.
patches = [
    (
        'CONFIG_HASH = hashlib.sha256(\n    json.dumps(asdict(CFG), sort_keys=True).encode("utf-8")\n).hexdigest()[:12]\n',
        'RUN_SIGNATURE = {\n    "config": asdict(CFG),\n    "source_commit": SOURCE_COMMIT,\n    "source_sha256": SOURCE_SHA256,\n}\nCONFIG_HASH = hashlib.sha256(\n    json.dumps(RUN_SIGNATURE, sort_keys=True).encode("utf-8")\n).hexdigest()[:12]\n',
        "source-aware run identity",
    ),
    (
        '    "Perforation", "Appendicular_Abscess", "Abscess_Location",\n',
        '    "Peritonitis", "Perforation", "Appendicular_Abscess", "Abscess_Location",\n',
        "Peritonitis leakage exclusion",
    ),
    (
        '        image_dim = int(self.backbone.fc.in_features)\n        self.backbone.fc = nn.Identity()\n        self.quality_pool = QualityAwarePool(image_dim)\n        self.image_proj = nn.Sequential(nn.Linear(image_dim, CFG.d_model), nn.LayerNorm(CFG.d_model), nn.GELU())\n',
        '        self.image_dim = int(self.backbone.fc.in_features)\n        self.backbone.fc = nn.Identity()\n        self._backbone_trainable = True\n        self.quality_pool = QualityAwarePool(self.image_dim)\n        self.image_proj = nn.Sequential(nn.Linear(self.image_dim, CFG.d_model), nn.LayerNorm(CFG.d_model), nn.GELU())\n',
        "explicit backbone state",
    ),
    (
        '    def set_backbone_trainable(self, trainable: bool) -> None:\n        for param in self.backbone.parameters():\n            param.requires_grad = trainable\n\n    def forward(\n',
        '    def train(self, mode: bool = True) -> "CGRMAT":\n        super().train(mode)\n        if mode and not self._backbone_trainable:\n            self.backbone.eval()\n        return self\n\n    def set_backbone_trainable(self, trainable: bool) -> None:\n        self._backbone_trainable = trainable\n        for param in self.backbone.parameters():\n            param.requires_grad = trainable\n        if not trainable:\n            self.backbone.eval()\n\n    def forward(\n',
        "frozen-backbone BatchNorm protection",
    ),
    (
        '        flat = images.view(batch * views, channels, height, width)\n        view_features = self.backbone(flat).view(batch, views, -1)\n',
        '        flat = images.view(batch * views, channels, height, width)\n        flat_mask = view_mask.reshape(-1)\n        view_features_flat = torch.zeros(\n            (batch * views, self.image_dim), device=images.device, dtype=images.dtype\n        )\n        if bool(flat_mask.any()):\n            view_features_flat[flat_mask] = self.backbone(flat[flat_mask])\n        view_features = view_features_flat.view(batch, views, self.image_dim)\n',
        "masked-view backbone execution",
    ),
]

for old, new, label in patches:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Safety patch '{label}' expected one target, found {count}.")
    source = source.replace(old, new, 1)
    print(f"  ✓ safety patch applied: {label}")

patched_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
if patched_sha != PATCHED_COMBINED_SHA256:
    raise RuntimeError(
        "Patched CGR-MAT source integrity failure.\n"
        f"Expected: {PATCHED_COMBINED_SHA256}\nActual:   {patched_sha}"
    )

os.environ["CGR_MAT_SOURCE_COMMIT"] = SOURCE_COMMIT
os.environ["CGR_MAT_SOURCE_SHA256"] = PATCHED_COMBINED_SHA256
print("✓ Patched CGR-MAT source integrity verified.")
print("Launching the pipeline...")
exec(compile(source, f"cgr_mat_pipeline@{SOURCE_COMMIT}", "exec"), globals(), globals())
