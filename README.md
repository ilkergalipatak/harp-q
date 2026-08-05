# HARP-Q

HARP-Q (Hierarchy-Aware Residual Progressive Quantization), büyük dil modeli ağırlıklarını tek
seferde GPU belleğine yüklemek yerine GPU–RAM–SSD hiyerarşisinde kademeli olarak sunmayı araştıran
bir prototiptir. Ağırlıklar `128 x 128` tile'lara ve sıralı `2+2+2+2` bit residual katmanlarına
ayrılır. Base katman her zaman hesaplanabilir; scheduler zaman ve cache bütçesi elverdikçe residual
katmanları ekler.

Bu depo şimdilik taşınabilir NumPy reference implementasyonunu hedefler. Mac sonuçları yalnızca
fonksiyonel doğruluk içindir. CUDA, gerçek async NVMe I/O ve NVIDIA performans ölçümleri Linux
aşamasına aittir.

## Hızlı başlangıç

Python 3.11+ gerekir. Doğrulanmış bağımlılık `requirements/runtime.lock` içinde sabitlenmiştir.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements/runtime.lock
make test PYTHON=.venv/bin/python
make demo PYTHON=.venv/bin/python
```

Codex Desktop'ın bu Mac'teki paketli Python ortamıyla kurulum yapmadan test etmek için:

```sh
make test PYTHON=/Users/ilkergalipatak/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
```

Tek bir NumPy linear katmanını calibration aktivasyonlarıyla quantize edip pack'lemek:

```sh
PYTHONPATH=src python -m harpq quantize \
  --weights weights.npy \
  --activations activations.npy \
  --matrix-id layers.0.attention.query \
  --output artifacts/query.harpq

PYTHONPATH=src python -m harpq inspect artifacts/query.harpq --verify-payloads
PYTHONPATH=src python -m harpq simulate artifacts/query.harpq \
  --capacity-bytes 1048576 --output artifacts/trace.json
PYTHONPATH=src python -m harpq replay artifacts/trace.json
```

Linux host'a geçildiğinde ilk pasif profil ve açıkça sınırlandırılmış aktif profil:

```sh
PYTHONPATH=src python -m harpq profile-host --output artifacts/host.json
PYTHONPATH=src python -m harpq profile-host --active \
  --scratch /path/on/target/nvme --output artifacts/host-active.json
```

Aktif profil 64 MiB RAM copy testi, geçici 256 MiB `fio + io_uring + O_DIRECT` testi ve sistemde
kuruluysa CUDA `deviceQuery`/`bandwidthTest` örneklerini çalıştırır. Geçici fio dizini sonunda
silinir.

## Tasarım sınırı

```text
offline: FP weights -> calibrate -> progressive quantize -> allocate -> aligned pack files
runtime: SSD packets -> host cache -> device cache -> progressive matmul
                             ^             |
                             +-- deadline-aware scheduler + deterministic trace
```

Araştırma kapsamı ve iddialar için `docs/RESEARCH_PLAN.md`, önceki çalışmalar için
`docs/PRIOR_ART.md`, mimari kararlar için `docs/DECISIONS.md`, makine geçişi için
`docs/HANDOFF.md` dosyalarına bakın.
