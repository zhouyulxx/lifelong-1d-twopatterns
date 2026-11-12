"""Import UCI Epileptic Seizure Recognition without quantization or synthetic replacement.

Exit with nonzero code if any of the 178 signal columns contain non-integer
values: this project requires native integer-valued one-dimensional sequences.
"""
import csv
import gzip
import hashlib
import io
import json
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.request import Request, urlopen

URLS=[
 "https://raw.githubusercontent.com/marakaspers/seizure-prediction/master/Epileptic_Seizure_Recognition.csv",
 "https://raw.githubusercontent.com/marakaspers/seizure-prediction/main/Epileptic_Seizure_Recognition.csv",
 "https://raw.githubusercontent.com/QiuyiWu/Epileptic-Seizure-Recognition-Data/master/A%26B%26C%26D%26E.csv",
]
OUT=Path("data/eeg_integer")
OUT.mkdir(parents=True,exist_ok=True)

def download():
    last=None
    for url in URLS:
        try:
            with urlopen(Request(url,headers={"User-Agent":"Mozilla/5.0"}),timeout=120) as r:
                data=r.read()
            if len(data)<1_000_000: raise ValueError("Download suspiciously small")
            print("Downloaded bytes",len(data),"from",url,flush=True)
            return url,data
        except Exception as e:
            print("Download failed:",str(e),flush=True)
            last=e
    raise RuntimeError(f"Could not fetch official UCI data: {last}")

def main():
    url, raw=download()
    content=raw.decode("utf-8-sig")
    reader=csv.DictReader(io.StringIO(content))
    if reader.fieldnames is None:raise ValueError("Missing CSV header")
    names=[f"X{i}" for i in range(1,179)]
    if not set(names+["y"]).issubset(reader.fieldnames):
        raise ValueError(f"Missing X1..X178 or y; head={reader.fieldnames[:8]}")
    classes=Counter()
    n=0;minv=None;maxv=None;fractional=0;bad_samples=[]
    for record in reader:
        n+=1
        classes[record["y"]]+=1
        for name in names:
            try: val=Decimal(record[name])
            except (InvalidOperation,TypeError):
                raise ValueError(f"Non-numeric signal: row={n} feature={name}")
            if not val.is_finite():raise ValueError(f"Nonfinite signal: row={n} feature={name}")
            if val != val.to_integral_value():
                fractional+=1
                if len(bad_samples)<8:bad_samples.append([n,name,str(val)])
            if minv is None or val<minv:minv=val
            if maxv is None or val>maxv:maxv=val
    meta={
        "dataset":"UCI Epileptic Seizure Recognition (id 388)",
        "source":url,"original_catalog":"https://archive.ics.uci.edu/dataset/388/epileptic+seizure+recognition","source_is_public_mirror":True,"sha256":hashlib.sha256(raw).hexdigest(),
        "source_format":"CSV with X1..X178 signals and y labels",
        "n_sequences":n,"sequence_length":178,
        "class_counts":dict(sorted(classes.items())),
        "value_min":str(minv),"value_max":str(maxv),
        "fractional_value_count":fractional,"fractional_examples":bad_samples,
        "all_input_signal_values_integer":fractional==0,
        "source_is_preprocessed_EEG":True,
        "claim":"integer-valued source samples only if all_input_signal_values_integer is true"
    }
    OUT.joinpath("SOURCE.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(meta,indent=2),flush=True)
    if n!=11500 or len(classes)!=5:
        raise ValueError("Unexpected sample count or class count")
    if fractional:
        raise ValueError("Source has fractional input samples; refusing to quantize")
    # Original source unchanged except gzip lossless compression.
    with gzip.open(OUT/"data.csv.gz","wb",compresslevel=6) as f: f.write(raw)
    print("Saved lossless original UCI source as gzip.",flush=True)

if __name__=="__main__":main()
