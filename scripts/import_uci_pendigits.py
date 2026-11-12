"""Fetch original UCI PenDigits training/test archives; no rounding or rebucketing."""
import csv,gzip,hashlib,io,json,zipfile
from collections import Counter
from pathlib import Path
from urllib.request import Request,urlopen
UCI_ZIP="https://archive.ics.uci.edu/static/public/81/pen+based+recognition+of+handwritten+digits.zip"
OUT=Path("data/pendigits")
OUT.mkdir(parents=True,exist_ok=True)

def get_zip():
    with urlopen(Request(UCI_ZIP,headers={"User-Agent":"Mozilla/5.0"}),timeout=90) as r:
        raw=r.read()
    if not zipfile.is_zipfile(io.BytesIO(raw)):
        raise RuntimeError("UCI source is not a valid ZIP")
    return raw

def read_archive(name,raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        matched=[n for n in z.namelist() if n.lower().endswith("/"+name) or n.lower()==name]
        if len(matched)!=1: raise RuntimeError(f"Cannot locate {name} in UCI archive; list={z.namelist()}")
        return z.read(matched[0])

def validate(raw,expected_rows):
    text=raw.decode("ascii")
    rows=[]
    for n,line in enumerate(csv.reader(io.StringIO(text)),1):
        if not line:continue
        if len(line)!=17:raise RuntimeError(f"Malformed source row {n} (expected 16 features + label)")
        row=[int(s.strip()) for s in line]
        if not all(0<=v<=100 for v in row[:16]):raise RuntimeError("Noninteger or out of range")
        if not 0<=row[16]<=9:raise RuntimeError("Invalid digit label")
        rows.append(row)
    if len(rows)!=expected_rows:raise RuntimeError(f"Expected {expected_rows} rows, got {len(rows)}")
    return rows

def main():
    raw=get_zip()
    print("Downloaded official UCI ZIP",len(raw),"bytes",flush=True)
    metadata={"dataset":"UCI Pen-Based Recognition of Handwritten Digits (ID 81)",
              "official_source":UCI_ZIP,
              "original_zip_sha256":hashlib.sha256(raw).hexdigest(),
              "has_native_integer_values":True,
              "feature_details":"16 ordered integers; 8 (x,y) pen-trajectory points, alternating channels",
              "feature_value_range":[0,100],"label_range":[0,9],
              "splits":{}}
    for name,n in (("pendigits.tra",7494),("pendigits.tes",3498)):
        data=read_archive(name,raw)
        values=validate(data,n)
        with gzip.open(OUT/f"{name}.gz","wb",compresslevel=7) as f:f.write(data)
        counts=dict(sorted(Counter(row[-1] for row in values).items()))
        metadata["splits"][name]={"n":n,"sha256":hashlib.sha256(data).hexdigest(),
                                  "class_counts":counts}
    (OUT/"SOURCE.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps(metadata,indent=2),flush=True)
if __name__=="__main__":main()
