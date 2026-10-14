"""Deterministic raw-media feature extraction and auditable CTC alignment."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import unicodedata

import av
import cv2
import imageio_ffmpeg
import librosa
import mediapipe as mp
import numpy as np
import opensmile
import pandas as pd
import pysptk
import pyworld
import torch
import yaml
from num2words import num2words
from scipy.spatial.transform import Rotation
from transformers import AutoModel, AutoTokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT.parent
SOURCE_DATA = PROJECT / 'data/source_data'
DATA = SOURCE_DATA / 'attachment1_raw_multimodal/mosei_raw_videos_100'
MODELS = ROOT / 'models/pretrained'
CONFIG=yaml.safe_load((Path(__file__).parent/'config.yaml').read_text())
FIXED_PROTOCOL=dict(seed=2026,torch_threads=4,sample_rate=16000,audio_hop=160,audio_window=400,
    video_fps=25,aligned_positions=50,unaligned_positions=500,face_detection_threshold=.5,
    face_presence_threshold=.5,alignment_confidence_warning=.5,
    bert_revision='86b5e0934494bd15c9632b12f734a8a67f723594',ctc_revision='22aad52d435eb6dbaf354bdad9b0da84ce7d6156',
    visual_schema='mediapipe_27_blendshapes_3_pose_5_geometry_v1',
    interface_schema='bert_wordpiece_50_with_explicit_masks_v1',audio_vision_dtype='float32',text_dtype='float16')
assert CONFIG==FIXED_PROTOCOL, 'config.yaml differs from the implemented fixed feature protocol; update implementation and rerun explicitly'
BLENDSHAPES = ['browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft',
 'browOuterUpRight', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft',
 'eyeBlinkRight', 'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight',
 'jawOpen', 'mouthSmileLeft', 'mouthSmileRight', 'mouthFrownLeft', 'mouthFrownRight',
 'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthStretchLeft',
 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight']
VISION_NAMES = BLENDSHAPES + ['pose_x_rad', 'pose_y_rad', 'pose_z_rad',
 'face_area_ratio', 'EAR', 'MAR', 'pose_speed_rad_s', 'landmarks_in_frame_ratio']
SMILE_QUALITY = ['HNRdBACF_sma3nz', 'jitterLocal_sma3nz', 'shimmerLocaldB_sma3nz',
 'logRelF0-H1-H2_sma3nz', 'alphaRatio_sma3', 'hammarbergIndex_sma3']
SMILE_FORMANT = [f'F{i}{kind}_sma3nz' for kind in ['frequency', 'bandwidth'] for i in [1, 2, 3]]
AUDIO_NAMES = ['F0_Hz', 'F0_semitone_27.5Hz', 'delta_F0_semitone', 'voiced',
 'RMS', 'delta_RMS', 'Loudness_sma3', 'ZCR'] + SMILE_QUALITY + [
 'spectral_centroid_Hz', 'spectral_bandwidth_Hz', 'spectral_rolloff85_Hz',
 'spectral_flux', 'spectral_flatness', 'slope0-500_sma3', 'slope500-1500_sma3',
 'spectral_entropy'] + SMILE_FORMANT + [f'MFCC{i}' for i in range(1,14)] + [
 f'delta_MFCC{i}' for i in range(1,14)] + [f'MCEP{i}' for i in range(20)]

def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))

def records():
    if not (DATA / 'label-100.xlsx').is_file():
        raise FileNotFoundError(f'Missing raw-media labels: {DATA / "label-100.xlsx"}')
    rows = pd.read_excel(DATA / 'label-100.xlsx', sheet_name='label').to_dict('records')
    for row in rows:
        row['clip_id'] = str(int(row['clip_id']))
        row['sample_id'] = f"{row['video_id']}$_${row['clip_id']}"
        row['label'] = float(row['label'])
        row['path'] = (DATA / row['video_id'] / (row['clip_id'] + '.mp4')).relative_to(PROJECT).as_posix()
    assert len(rows) == 100 and len({r['sample_id'] for r in rows}) == 100
    assert {PROJECT / r['path'] for r in rows} == set(DATA.rglob('*.mp4'))
    return rows

def probe(path):
    with av.open(str(path)) as c:
        v, a = c.streams.video[0], c.streams.audio[0]
        vs, aus = float((v.start_time or 0)*v.time_base), float((a.start_time or 0)*a.time_base)
        vd, ad = float(v.duration*v.time_base), float(a.duration*a.time_base)
        start, end = max(vs, aus), min(vs+vd, aus+ad)
        return dict(T_video=vd, T_audio=ad, video_start=vs, audio_start=aus,
            common_start=start, duration=end-start, fps=float(v.average_rate),
            source_frames=v.frames, width=v.width, height=v.height,
            source_sr=a.sample_rate, source_channels=a.codec_context.channels)

def waveform(path, meta):
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i', str(path),
        '-vn', '-ac', '1', '-ar', '16000', '-f', 'f32le', 'pipe:1']
    y = np.frombuffer(subprocess.check_output(command), dtype='<f4').copy()
    start = round((meta['common_start']-meta['audio_start'])*16000)
    count = int(np.floor(meta['duration']*16000))
    y = y[start:start+count]
    # Container duration and resampler rounding can differ by a few samples.
    # Keep only actual decoded samples; never synthesize a missing tail.
    assert len(y)>400, (len(y), count)
    return y

def normalized(word):
    s = unicodedata.normalize('NFKD', word.replace('’', "'")).encode('ascii', 'ignore').decode()
    s = re.sub(r'\d+(?:\.\d+)?', lambda m: num2words(m.group()).replace('-', ' '), s)
    return re.sub("[^A-Z']", '', s.upper()).strip("'")

def word_items(text):
    items = []
    for match in re.finditer(r'\S+', text):
        w = normalized(match.group())
        if w: items.append(dict(i=len(items), word=match.group(), normalized=w,
                               char_start=match.start(), char_end=match.end()))
    return items

def ctc_viterbi(logp, tokens, blank=0):
    """Exact CTC topology, including repeat-label constraint; global endpoints."""
    T, _ = logp.shape
    extended = np.full(2*len(tokens)+1, blank, dtype=np.int64)
    extended[1::2] = tokens
    S = len(extended)
    skip = np.zeros(S, bool)
    skip[2:] = (extended[2:] != blank) & (extended[2:] != extended[:-2])
    score = np.full(S, -np.inf, np.float32)
    score[0] = logp[0, blank]
    score[1] = logp[0, tokens[0]]
    back = np.zeros((T, S), np.uint8)
    for t in range(1,T):
        one = np.r_[-np.inf, score[:-1]]
        two = np.r_[-np.inf, -np.inf, score[:-2]]
        two[~skip] = -np.inf
        candidates = np.stack([score, one, two])
        back[t] = candidates.argmax(axis=0)
        score = candidates.max(axis=0) + logp[t, extended]
    state = S-1 if score[-1] >= score[-2] else S-2
    if not np.isfinite(score[state]): raise ValueError('No valid CTC path')
    path = np.empty(T, np.int32)
    for t in range(T-1,-1,-1):
        path[t] = state
        state -= int(back[t,state])
    assert path[0] in (0,1)
    return path

def wer(reference, hypothesis):
    a, b = reference.split(), hypothesis.split()
    dp = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1): nxt.append(min(nxt[-1]+1, dp[j]+1, dp[j-1]+(x!=y)))
        dp=nxt
    return dp[-1]/max(len(a),1)

class TextExtractor:
    def __init__(self, threads=4):
        torch.set_num_threads(threads)
        torch.manual_seed(2026)
        torch.use_deterministic_algorithms(True)
        self.tokenizer = AutoTokenizer.from_pretrained(MODELS/'bert', local_files_only=True)
        self.bert = AutoModel.from_pretrained(MODELS/'bert', local_files_only=True).eval()
        self.processor = Wav2Vec2Processor.from_pretrained(MODELS/'ctc', local_files_only=True)
        self.ctc = Wav2Vec2ForCTC.from_pretrained(MODELS/'ctc', local_files_only=True).eval()

    def extract(self, text, y):
        words = word_items(text)
        transcript = '|'.join(w['normalized'] for w in words)
        vocab = self.processor.tokenizer.get_vocab()
        tokens = [vocab[c] for c in transcript]
        values = self.processor(y, sampling_rate=16000, return_tensors='pt').input_values
        with torch.inference_mode(): logp = self.ctc(values).logits[0].log_softmax(-1).numpy()
        path = ctc_viterbi(logp, tokens, self.processor.tokenizer.pad_token_id)
        pos = 0
        for w in words:
            indices = np.flatnonzero((path >= 2*pos+1) & (path <= 2*(pos+len(w['normalized'])-1)+1))
            char_frames = indices[path[indices] % 2 == 1]
            # wav2vec2's 400-sample receptive field, 320-sample stride.
            start_sample = int(indices[0])*320+40
            end_sample = min(len(y), int(indices[-1])*320+360)
            w.update(s=start_sample/16000, e=end_sample/16000,
                     s_sample=start_sample, e_sample=end_sample,
                     confidence=float(np.exp(logp[char_frames, np.array(tokens)[(path[char_frames]-1)//2]]).mean()),
                     ctc_frames=[int(indices[0]),int(indices[-1])+1])
            pos += len(w['normalized'])+1
        enc = self.tokenizer(text, return_offsets_mapping=True, return_tensors='pt')
        offsets = enc.pop('offset_mapping')[0].numpy()
        assert enc.input_ids.shape[1] <= 512, 'Long-context chunking required'
        with torch.inference_mode(): hidden = self.bert(**enc).last_hidden_state[0].numpy()
        token_word = np.full(len(offsets), -1, np.int32)
        vectors=[]
        for w in words:
            ids = np.flatnonzero((offsets[:,1] > w['char_start']) & (offsets[:,0] < w['char_end']) & (offsets[:,1]>offsets[:,0]))
            assert len(ids)>0
            w['bert_subtokens']=ids.tolist()
            token_word[ids]=w['i']
            vectors.append(hidden[ids].mean(axis=0))
        hyp = self.processor.decode(logp.argmax(axis=-1))
        return words, dict(text=np.array(vectors, np.float32), token_hidden=hidden,
            token_ids=enc.input_ids[0].numpy(), token_word=token_word, token_offsets=offsets), dict(
            align_method='wav2vec2_ctc_viterbi', asr_hypothesis=hyp,
            wer_ctc=wer(transcript.replace('|',' '), hyp), align_conf=float(np.mean([w['confidence'] for w in words])))

def audio_features(y):
    yd = y.astype(np.float64)
    frames = librosa.util.frame(y, frame_length=400, hop_length=160).T
    times = (np.arange(len(frames))*160+200)/16000
    # openSMILE's ctypes options are ASCII-only; keep original configs unchanged.
    config_root=Path(opensmile.__file__).parent/'core/config'
    # Isolate temporary links by environment so a relocated project cannot
    # accidentally reuse a stale link from a different Python installation.
    config_key=hashlib.sha256(str(config_root.resolve()).encode()).hexdigest()[:12]
    ascii_root=Path('/tmp') / f'coling2027-opensmile-{config_key}'
    if not ascii_root.exists():ascii_root.symlink_to(config_root,target_is_directory=True)
    assert ascii_root.resolve()==config_root.resolve()
    class AsciiSmile(opensmile.Smile):
        @property
        def default_config_root(self):return str(ascii_root)
    smile = AsciiSmile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                           feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
    sdf = smile.process_signal(y, 16000)
    st = (sdf.index.get_level_values('start').total_seconds().to_numpy()+.01)
    # Expose the source timestamps: never concatenate packages by frame number.
    def sm(name): return np.interp(times, st, sdf[name].to_numpy())
    f0, wt = pyworld.dio(yd, 16000, frame_period=10, f0_floor=50, f0_ceil=600)
    f0 = pyworld.stonemask(yd, f0, wt, 16000)
    spec = pyworld.cheaptrick(yd, f0, wt, 16000)
    mc = pysptk.sp2mc(spec, order=19, alpha=.42)
    nearest = np.abs(wt[:,None]-times[None,:]).argmin(axis=0)
    pitch = f0[nearest]; voiced = pitch>0
    semitone=np.zeros_like(pitch);semitone[voiced]=12*np.log2(pitch[voiced]/27.5)
    dp=np.r_[0.,np.diff(semitone)];dp[~(voiced & np.r_[False,voiced[:-1]])]=0
    rms=np.sqrt(np.mean(frames**2,axis=1)); zcr=np.mean(np.diff(np.signbit(frames),axis=1),axis=1)
    S=np.abs(librosa.stft(y,n_fft=400,hop_length=160,win_length=400,center=False))
    power=S*S
    centroid=librosa.feature.spectral_centroid(S=S,sr=16000)[0]
    bandwidth=librosa.feature.spectral_bandwidth(S=S,sr=16000)[0]
    roll=librosa.feature.spectral_rolloff(S=S,sr=16000,roll_percent=.85)[0]
    norm=S/(np.linalg.norm(S,axis=0,keepdims=True)+1e-12)
    flux=np.r_[0., np.sqrt(np.sum(np.diff(norm,axis=1)**2,axis=0))]
    flat=librosa.feature.spectral_flatness(S=S)[0]
    prob=power/(power.sum(axis=0,keepdims=True)+1e-12)
    entropy=-np.sum(prob*np.log(prob+1e-12),axis=0)/np.log(len(prob))
    mel=librosa.feature.melspectrogram(S=power,sr=16000,n_fft=400,n_mels=40)
    mfcc=librosa.feature.mfcc(S=librosa.power_to_db(mel),n_mfcc=14)[1:14]
    delta=librosa.feature.delta(mfcc,width=9,mode='nearest')
    features=np.column_stack([pitch,semitone,dp,voiced,rms,np.r_[0.,np.diff(rms)],
        sm('Loudness_sma3'),zcr,*[sm(k) for k in SMILE_QUALITY],centroid,bandwidth,
        roll,flux,flat,sm('slope0-500_sma3'),sm('slope500-1500_sma3'),entropy,
        *[sm(k) for k in SMILE_FORMANT],mfcc.T,delta.T,mc[nearest]])
    nonfinite=int((~np.isfinite(features)).sum())
    features=np.nan_to_num(features,nan=0,posinf=0,neginf=0).astype(np.float32)
    assert features.shape==(len(times),74)
    return features, times, dict(audio_nonfinite_replaced=nonfinite, opensmile_frames=len(sdf), world_frames=len(wt))

def sampled_frames(path, meta):
    times = np.arange(int(np.ceil(meta['duration']*25)))/25
    times = times[times<meta['duration']]
    with av.open(str(path)) as c:
        source=[]
        for i, f in enumerate(c.decode(video=0)):
            pts=float(f.pts*f.time_base)-meta['common_start']
            # Resize with a fixed aspect-preserving rule for inference and replay.
            width=min(640,f.width);height=round(f.height*width/f.width)
            rgb=f.reformat(width=width,height=height,format='rgb24').to_ndarray()
            source.append((pts,i,rgb))
    pts=np.array([f[0] for f in source])
    indices=np.abs(pts[:,None]-times[None,:]).argmin(axis=0)
    for t,index in zip(times,indices):
        yield float(t),source[index][0],source[index][1],source[index][2]

def vision_features(path, meta):
    options=mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODELS/'face_landmarker.task')),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=5,
        min_face_detection_confidence=.5,min_face_presence_confidence=.5,
        output_face_blendshapes=True,output_facial_transformation_matrixes=True)
    result=[]; times=[]; source_pts=[]; source_idx=[]; masks=[]; nfaces=[]; boxes=[]
    previous_box=None;previous_rotation=None;track_switches=0
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as model:
        for t,pts,idx,rgb in sampled_frames(path,meta):
            pred=model.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb)))
            times.append(t);source_pts.append(pts);source_idx.append(idx)
            nfaces.append(len(pred.face_landmarks))
            value=np.zeros(35,np.float32);box=np.zeros(4,np.float32)
            if pred.face_landmarks:
                points=[np.array([[p.x,p.y,p.z] for p in face]) for face in pred.face_landmarks]
                bb=[np.r_[p[:,:2].min(axis=0),p[:,:2].max(axis=0)] for p in points]
                areas=np.array([np.prod(b[2:]-b[:2]) for b in bb]);choice=int(areas.argmax())
                if previous_box is not None:
                    intersect=np.array([np.maximum(0,np.minimum(b[2:],previous_box[2:])-np.maximum(b[:2],previous_box[:2])).prod() for b in bb])
                    iou=intersect/(areas+np.prod(previous_box[2:]-previous_box[:2])-intersect+1e-12)
                    if iou.max()>=.5: choice=int(iou.argmax())
                    else: track_switches+=1
                p=points[choice];box=bb[choice];previous_box=box
                xy=p[:,:2]*[rgb.shape[1],rgb.shape[0]]
                dist=lambda a,b:np.linalg.norm(xy[a]-xy[b])
                ear=((dist(160,144)+dist(158,153))/(2*dist(33,133)+1e-9)+(dist(385,380)+dist(387,373))/(2*dist(362,263)+1e-9))/2
                mar=dist(13,14)/(dist(78,308)+1e-9)
                rotation=Rotation.from_matrix(pred.facial_transformation_matrixes[choice][:3,:3])
                pose=rotation.as_euler('xyz')
                speed=0 if previous_rotation is None else (previous_rotation.inv()*rotation).magnitude()*25
                previous_rotation=rotation
                blend={x.category_name:x.score for x in pred.face_blendshapes[choice]}
                value[:27]=[blend[k] for k in BLENDSHAPES]
                value[27:30]=pose
                value[30:]=[areas[choice],ear,mar,speed,float(((p[:,:2]>=0)&(p[:,:2]<=1)).all(axis=1).mean())]
                masks.append(True)
            else:
                masks.append(False);previous_rotation=None
            result.append(value);boxes.append(box)
    return dict(vision=np.array(result),vision_times=np.array(times),face_mask=np.array(masks),
        source_video_pts=np.array(source_pts),source_video_index=np.array(source_idx,np.int32),
        face_boxes=np.array(boxes),n_faces=np.array(nfaces,np.int16)), dict(track_switches=track_switches,n_faces_max=max(nfaces))

def word_sample_bounds(word):
    """Word boundaries on the 16 kHz clock; accept older time-only records."""
    return (int(word.get('s_sample', round(word['s']*16000))),
            int(word.get('e_sample', round(word['e']*16000))))


def interval_frame_indices(times, start_sample, end_sample):
    """Half-open membership for the audio/video grids on the common sample clock."""
    ticks = np.rint(np.asarray(times)*16000).astype(np.int64)
    return np.flatnonzero((ticks >= start_sample) & (ticks < end_sample))


def pool_words(words,audio,at,vision,vt,face_mask):
    aa=[];vv=[];masks=[];available=[];alignment=[];fallback=[]
    for w in words:
        start, end = word_sample_bounds(w)
        ai=interval_frame_indices(at,start,end)
        vi=interval_frame_indices(vt,start,end)
        af=vf=False
        # Compare doubled integer centers; exact midpoint ties choose the first frame.
        if not len(ai):ai=np.array([np.abs(2*np.rint(at*16000).astype(np.int64)-start-end).argmin()]);af=True
        if not len(vi):vi=np.array([np.abs(2*np.rint(vt*16000).astype(np.int64)-start-end).argmin()]);vf=True
        valid=vi[face_mask[vi]]
        aa.append(audio[ai].mean(axis=0));vv.append(vision[valid].mean(axis=0) if len(valid) else np.zeros(35))
        usable=w.get('timing_usable',w['confidence']>=.5)
        available.append([True,True,bool(len(valid))])
        masks.append([True,usable,bool(len(valid)) and usable]);alignment.append(usable);fallback.append([af,vf])
        w.update(a_frames=ai.tolist(),v_frames=vi.tolist(),v_valid_frames=valid.tolist(),
                 audio_fallback=af,video_fallback=vf)
    return dict(audio=np.array(aa,np.float32),vision=np.array(vv,np.float32),
                mask=np.array(masks,bool),availability_mask=np.array(available,bool),
                alignment_mask=np.array(alignment,bool),fallback=np.array(fallback,bool))
