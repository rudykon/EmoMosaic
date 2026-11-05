"""Persistent local Qwen3.5 worker, isolating Transformers from older BERT runs."""
import argparse
import atexit
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from project_paths import python_executable
from data import ROOT


class Qwen35Worker:
    def __init__(self, encoder, folder, device='cpu', threads=4):
        python = python_executable('qwen35')
        self.directory = tempfile.TemporaryDirectory(prefix='emomosaic-qwen35-')
        self.log = (Path(self.directory.name)/'worker.log').open('w+')
        self.process = subprocess.Popen(
            [str(python), str(Path(__file__).resolve()), '--encoder', encoder,
             '--folder', str(folder), '--device', str(device), '--threads', str(threads)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True,
            env=dict(os.environ, TOKENIZERS_PARALLELISM='false'))
        atexit.register(self.close)
        self._receive()

    def _receive(self):
        line = self.process.stdout.readline()
        if not line:
            self.log.flush(); self.log.seek(0)
            raise RuntimeError('Qwen3.5 worker failed: '+self.log.read()[-5000:])
        value = json.loads(line)
        if 'error' in value:
            raise RuntimeError(value['error'])
        return value

    def encode(self, texts, layer):
        target = Path(self.directory.name)/'response.npz'
        self.process.stdin.write(json.dumps({'texts': texts, 'layer': int(layer), 'output': str(target)})+'\n')
        self.process.stdin.flush()
        self._receive()
        with np.load(target) as z:
            return {key: z[key] for key in z.files}

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)
        if not self.log.closed:
            self.log.close()
        self.directory.cleanup()


def main():
    import torch
    from transformers import AutoTokenizer
    from text_encoder_registry import load_frozen_text_model
    ap=argparse.ArgumentParser()
    ap.add_argument('--encoder',required=True)
    ap.add_argument('--folder',type=Path,required=True)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--threads',type=int,default=4)
    args=ap.parse_args();torch.set_num_threads(args.threads)
    tokenizer=AutoTokenizer.from_pretrained(args.folder,local_files_only=True)
    tokenizer.pad_token=tokenizer.pad_token or tokenizer.eos_token
    model=load_frozen_text_model(args.encoder,args.folder,args.device)
    print(json.dumps({'ready': True}),flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        texts=[t if t.strip() else tokenizer.eos_token for t in request['texts']]
        encoded=tokenizer(texts,padding=True,truncation=False,add_special_tokens=False,
                          return_offsets_mapping=True,return_tensors='pt')
        offsets=encoded.pop('offset_mapping').numpy()
        batch={k:v.to(args.device) for k,v in encoded.items()}
        with torch.inference_mode():
            h=model(**batch,output_hidden_states=True,use_cache=False).hidden_states[request['layer']]
        h=h.float().cpu().numpy()
        if not np.isfinite(h).all() or np.abs(h).max()>65504:
            raise ValueError('Hidden values cannot be represented safely in float16')
        np.savez(request['output'],hidden=h.astype(np.float16),offsets=offsets,
                 attention_mask=encoded['attention_mask'].numpy())
        print(json.dumps({'complete': True}),flush=True)


if __name__=='__main__':
    main()
