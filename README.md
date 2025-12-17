
# AOL Comphys – Trash Classifier MVP (TrashNet + PyTorch + Streamlit)

MVP untuk tugas SDG 12 (Responsible Consumption and Production – Plastic Waste Management):

- Klasifikasi sampah dari gambar (TrashNet: cardboard, glass, metal, paper, plastic, trash)
- Output: label + confidence + top-2 + flag `needs_review` + instruksi buang
- UI: Streamlit (upload/camera)

> Catatan: Dataset dan model weights **tidak disimpan di GitHub** (besar). Jadi setelah clone, lakukan setup 1x.

---

Download Git -> Download Python 3.11 -> Buat Folder -> Paste di *POWER-SHELL* folder tersebut :

```bash
mkdir AOL_Comphys_Run
cd AOL_Comphys_Run
git clone https://github.com/LecyLecy/aol-comphys-group6-ll01-b28.git
cd aol-comphys-group6-ll01-b28
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install streamlit
cd .\data\trashnet
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip" -OutFile "dataset-resized.zip"
Expand-Archive -Force .\dataset-resized.zip .
$classes = @("cardboard","glass","metal","paper","plastic","trash")
foreach ($c in $classes) { New-Item -ItemType Directory -Force -Path ".\raw\$c" | Out-Null; Copy-Item -Force -Path ".\dataset-resized\$c\*" -Destination ".\raw\$c\" }
Remove-Item -Recurse -Force .\dataset-resized
cd ..\..
python -m src.ml.train
streamlit run app.py
```


## Requirements

- Windows 10/11
- Git
- Python **3.11.x** (recommended)
- (Opsional) GPU NVIDIA kalau ada, tapi CPU juga bisa

---

## Quick Start (Teman Tim)

### 1) Clone repo

```bash
git clone https://github.com/LecyLecy/aol-comphys-group6-ll01-b28.git
cd aol-comphys-group6-ll01-b28
```


### 2) Buat virtual environment (Python 3.11) + install dependencies

**Windows PowerShell**

<pre class="overflow-visible! px-0!" data-start="1469" data-end="1687"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="@w-xl/main:top-9 sticky top-[calc(--spacing(9)+var(--header-height))]"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-powershell"><span><span>py </span><span>-3</span><span>.</span><span>11</span><span></span><span>-m</span><span> venv .venv
.\.venv\Scripts\activate

python </span><span>-m</span><span> pip install </span><span>--upgrade</span><span> pip
pip install </span><span>-r</span><span> requirements.txt

</span><span># kalau belum ada streamlit di requirements, install manual:</span><span>
pip install streamlit
</span></span></code></div></div></pre>

Cek versi:

<pre class="overflow-visible! px-0!" data-start="1700" data-end="1734"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="@w-xl/main:top-9 sticky top-[calc(--spacing(9)+var(--header-height))]"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-powershell"><span><span>python </span><span>--version</span><span>
</span></span></code></div></div></pre>

### 3) Download dataset TrashNet (1x)

Jalankan dari root project:

<pre class="overflow-visible! px-0!" data-start="1802" data-end="2321"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="@w-xl/main:top-9 sticky top-[calc(--spacing(9)+var(--header-height))]"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-powershell"><span><span>cd</span><span> .\data\trashnet

</span><span>Invoke-WebRequest</span><span></span><span>-Uri</span><span></span><span>"https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip"</span><span></span><span>-OutFile</span><span></span><span>"dataset-resized.zip"</span><span>
</span><span>Expand-Archive</span><span></span><span>-Force</span><span> .\data</span><span>set-resized</span><span>.zip .

</span><span>$classes</span><span> = </span><span>@</span><span>(</span><span>"cardboard"</span><span>,</span><span>"glass"</span><span>,</span><span>"metal"</span><span>,</span><span>"paper"</span><span>,</span><span>"plastic"</span><span>,</span><span>"trash"</span><span>)
</span><span>foreach</span><span> (</span><span>$c</span><span></span><span>in</span><span></span><span>$classes</span><span>) {
  </span><span>New-Item</span><span></span><span>-ItemType</span><span> Directory </span><span>-Force</span><span></span><span>-Path</span><span></span><span>".\raw\$c</span><span>" | </span><span>Out-Null</span><span>
  </span><span>Copy-Item</span><span></span><span>-Force</span><span></span><span>-Path</span><span></span><span>".\dataset-resized\$c</span><span>\*" </span><span>-Destination</span><span></span><span>".\raw\$c</span><span>\"
}

</span><span>Remove-Item</span><span></span><span>-Recurse</span><span></span><span>-Force</span><span> .\data</span><span>set-resized</span><span>
</span><span>cd</span><span> ..\..
</span></span></code></div></div></pre>

Sanity check (harus keluar angka > 0):

<pre class="overflow-visible! px-0!" data-start="2362" data-end="2539"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="@w-xl/main:top-9 sticky top-[calc(--spacing(9)+var(--header-height))]"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-powershell"><span><span>python </span><span>-c</span><span></span><span>"from src.ml.config import Config; from src.ml.train import gather_samples; c=Config(); print('images:', len(gather_samples(c.data_dir, c.classes)))"</span><span>
</span></span></code></div></div></pre>

### 4) Train model (1x) untuk menghasilkan `models/model.pth`

<pre class="overflow-visible! px-0!" data-start="2603" data-end="2643"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="@w-xl/main:top-9 sticky top-[calc(--spacing(9)+var(--header-height))]"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-powershell"><span><span>python </span><span>-m</span><span> src.ml.train
</span></span></code></div></div></pre>

(Optional) Evaluasi + confusion matrix:

<pre class="overflow-visible! px-0!" data-start="2685" data-end="2724"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="@w-xl/main:top-9 sticky top-[calc(--spacing(9)+var(--header-height))]"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-powershell"><span><span>python </span><span>-m</span><span> src.ml.eval
</span></span></code></div></div></pre>

### 5) Run Streamlit UI

<pre class="overflow-visible! px-0!" data-start="2750" data-end="2788"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="@w-xl/main:top-9 sticky top-[calc(--spacing(9)+var(--header-height))]"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-powershell"><span><span>streamlit run app.py
</span></span></code></div></div></pre>

---

## Cara pakai tanpa UI (CLI test)

<pre class="overflow-visible! px-0!" data-start="2829" data-end="2967"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="@w-xl/main:top-9 sticky top-[calc(--spacing(9)+var(--header-height))]"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-powershell"><span><span>python </span><span>-c</span><span></span><span>"from inference.api import predict_path_ui; print(predict_path_ui(r'data/trashnet/raw/plastic/plastic1.jpg'))"</span><span>
</span></span></code></div></div></pre>

---

## Output Inference (untuk integrasi UI)

File: `inference/api.py`

Output contoh:

<pre class="overflow-visible! px-0!" data-start="3056" data-end="3327"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="@w-xl/main:top-9 sticky top-[calc(--spacing(9)+var(--header-height))]"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-json"><span><span>{</span><span>
  </span><span>"label"</span><span>:</span><span></span><span>"plastic"</span><span>,</span><span>
  </span><span>"confidence"</span><span>:</span><span></span><span>0.87</span><span>,</span><span>
  </span><span>"top"</span><span>:</span><span></span><span>[</span><span>{</span><span>"label"</span><span>:</span><span>"plastic"</span><span>,</span><span>"confidence"</span><span>:</span><span>0.87</span><span>}</span><span>,</span><span>{</span><span>"label"</span><span>:</span><span>"glass"</span><span>,</span><span>"confidence"</span><span>:</span><span>0.10</span><span>}</span><span>]</span><span>,</span><span>
  </span><span>"needs_review"</span><span>:</span><span></span><span>false</span><span></span><span>,</span><span>
  </span><span>"instruction"</span><span>:</span><span></span><span>"Bilas cepat, keringkan, lalu buang ke Recycle Plastik."</span><span>,</span><span>
  </span><span>"probs"</span><span>:</span><span></span><span>{</span><span></span><span>"..."</span><span>:</span><span></span><span>0.0</span><span></span><span>}</span><span>
</span><span>}</span><span>
</span></span></code></div></div></pre>

`needs_review` akan true jika:

* confidence rendah / margin top1-top2 kecil, atau
* pasangan kelas sering ketukar (mis. glass vs plastic) -> disarankan foto ulang / override manual.

---

## Project Structure (ringkas)

* `src/ml/` : training + eval (PyTorch)
* `models/` : `labels.json` (tracked), `model.pth` (generated locally)
* `inference/` : API prediksi untuk UI
* `app.py` : Streamlit UI
* `data/trashnet/raw/` : dataset lokal (tidak di-commit)

---

## Troubleshooting

**1) Model belum ada / UI error file not found**

* Pastikan sudah menjalankan:

<pre class="overflow-visible! px-0!" data-start="3885" data-end="3925"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="@w-xl/main:top-9 sticky top-[calc(--spacing(9)+var(--header-height))]"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-powershell"><span><span>python </span><span>-m</span><span> src.ml.train
</span></span></code></div></div></pre>

dan file `models/model.pth` sudah muncul.

**2) Python error type hints / versi**

* Pastikan Python 3.11 aktif:

<pre class="overflow-visible! px-0!" data-start="4038" data-end="4072"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="@w-xl/main:top-9 sticky top-[calc(--spacing(9)+var(--header-height))]"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-powershell"><span><span>python </span><span>--version</span><span>
</span></span></code></div></div></pre>

**3) Training lama di CPU**

* Normal. Dataset kecil (TrashNet) harusnya tetap bisa selesai.

<pre class="overflow-visible! px-0!" data-start="4166" data-end="4829" data-is-last-node=""><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="@w-xl/main:top-9 sticky top-[calc(--spacing(9)+var(--header-height))]"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>
---

</span><span>## Tambahan kecil biar makin mulus (recommended)</span><span>
</span><span>1</span><span>)</span><span></span><span>**Tambahin</span><span></span><span>Streamlit</span><span></span><span>ke</span><span></span><span>`requirements.txt`**</span><span></span><span>biar temenmu nggak install manual:</span><span>
   </span><span>-</span><span></span><span>Tambah baris:</span><span></span><span>`streamlit>=1.30`</span><span>

</span><span>2</span><span>)</span><span></span><span>**Jangan</span><span></span><span>simpan</span><span></span><span>zip</span><span></span><span>&</span><span></span><span>folder</span><span></span><span>aneh</span><span></span><span>ke</span><span></span><span>repo**</span><span>
</span><span>Kalau</span><span></span><span>kamu</span><span></span><span>lihat</span><span></span><span>`__MACOSX`</span><span></span><span>atau</span><span></span><span>`dataset-resized.zip`</span><span></span><span>di</span><span></span><span>folder,</span><span></span><span>itu</span><span></span><span>aman</span><span></span><span>karena</span><span></span><span>harusnya</span><span></span><span>ke-ignore.</span><span>

---

</span><span>Kalau</span><span></span><span>kamu</span><span></span><span>mau,</span><span></span><span>aku</span><span></span><span>juga</span><span></span><span>bisa</span><span></span><span>kasih</span><span></span><span>`.gitignore`</span><span></span><span>final yang rapi untuk ignore:</span><span>
</span><span>-</span><span></span><span>`data/`</span><span>
</span><span>-</span><span></span><span>`dataset-resized.zip`</span><span>
</span><span>-</span><span></span><span>`__MACOSX`</span><span>
</span><span>-</span><span></span><span>`models/model.pth`</span><span>
</span><span>-</span><span></span><span>cache</span><span></span><span>python</span><span>

</span><span>Tapi yang paling penting:</span><span></span><span>README</span><span></span><span>di</span><span></span><span>atas</span><span></span><span>sudah</span><span></span><span>cukup</span><span></span><span>bikin</span><span></span><span>temenmu</span><span></span><span>**clone</span><span></span><span>→</span><span></span><span>setup</span><span></span><span>→</span><span></span><span>train</span><span></span><span>→</span><span></span><span>run</span><span></span><span>UI**</span><span></span><span>tanpa</span><span></span><span>nanya-nanya</span><span></span><span>lagi.</span><span>
</span><span>::contentReference[oaicite:0]{index=0}</span></span></code></div></div></pre>
