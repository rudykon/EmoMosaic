"""Download only the pinned public encoders required by the exported model."""
import json
from huggingface_hub import snapshot_download
from data import ROOT
from improvement_common import digest


def main():
    entries=json.loads((ROOT/'models/rmnet_bert_lora/pretrained_manifest.json').read_text())
    for entry in entries:
        folder=ROOT/entry['local_dir']
        snapshot_download(repo_id=entry['repository'],revision=entry['revision'],local_dir=folder,
                          allow_patterns=['*.json','*.txt','*.model','*.safetensors','*.tiktoken'],max_workers=4)
        expected=entry['weights_sha256']
        if isinstance(expected,str):expected={'model.safetensors':expected}
        for name,value in expected.items():
            if digest(folder/name)!=value:raise ValueError(f'Pretrained SHA256 mismatch: {folder/name}')
        print(f'Verified {entry["repository"]}@{entry["revision"]}',flush=True)


if __name__=='__main__':main()
