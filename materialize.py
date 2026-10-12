"""Verify/unpack evidence or reassemble one adapter without model inference."""
from pathlib import Path
import argparse, gzip, hashlib, json, shutil
ROOT=Path(__file__).resolve().parent
def sha(data): return hashlib.sha256(data).hexdigest()
p=argparse.ArgumentParser()
p.add_argument("action",choices=["list","adapter","evidence"])
p.add_argument("name",nargs="?")
p.add_argument("--out",type=Path,default=ROOT/"outputs")
a=p.parse_args()
if a.action=="list":
    print((ROOT/"models/index.json").read_text())
elif a.action=="adapter":
    allowed={m["name"] for m in json.loads((ROOT/"models/index.json").read_text())}
    if a.name not in allowed: p.error("Choose a model name printed by list")
    source=ROOT/"models"/a.name
    meta=json.loads((source/"parts.json").read_text())
    dest=a.out/a.name
    dest.mkdir(parents=True,exist_ok=False)
    h=hashlib.sha256(); size=0
    with (dest/"adapter_model.safetensors").open("xb") as output:
        for part in meta["parts"]:
            data=(source/part["name"]).read_bytes()
            assert sha(data)==part["sha256"] and len(data)==part["bytes"],part["name"]
            output.write(data); h.update(data); size+=len(data)
    assert h.hexdigest()==meta["sha256"] and size==meta["source_bytes"]
    shutil.copyfile(source/"adapter_config.json",dest/"adapter_config.json")
    print("Verified adapter:",dest,"; model load/inference not performed")
else:
    index=json.loads((ROOT/"SOURCE_MANIFEST.json").read_text())
    selected=[r for r in index["files"] if r["path"].startswith("evidence/") and (not a.name or r["path"].startswith("evidence/"+a.name.rstrip("/")+"/"))]
    if not selected: p.error("No matching evidence group; use fixed_history, training, live_play, behavioral_sft, rl or configs")
    for r in selected:
        source=ROOT/r["path"]; packed=source.read_bytes()
        assert sha(packed)==r["release_sha256"]
        data=gzip.decompress(packed)
        assert sha(data)==r["release_uncompressed_sha256"]
        dest=a.out/Path(r["path"]).with_suffix("")
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open("xb") as f: f.write(data)
    print("Unpacked",len(selected),"verified evidence files under",a.out)
