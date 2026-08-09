# HARP-Q

HARP-Q (Hierarchy-Aware Residual Progressive Quantization), büyük dil modeli ağırlıklarını tek
seferde GPU belleğine yüklemek yerine GPU–RAM–SSD hiyerarşisinde kademeli olarak sunmayı araştıran
bir prototiptir. Ağırlıklar `128 x 128` tile'lara ve sıralı `2+2+2+2` bit residual katmanlarına
ayrılır. Base katman her zaman hesaplanabilir; scheduler zaman ve cache bütçesi elverdikçe residual
katmanları ekler.

NumPy uygulaması formatın ve sayısal davranışın taşınabilir referansıdır. C++17 katmanı aynı stage
payload'larını bağımsız olarak decode eder; CUDA doğruluk oracle'ı seçilen residual prefix'i GPU'da
açıp cuBLAS ile hesaplar. İlk batch-one fused GEMV ise FP32 ağırlık oluşturmadan device-resident
int2 code ve FP16 scale verisini doğrudan tüketir. Mac sonuçları yalnızca fonksiyonel doğruluk
içindir. Gerçek async NVMe I/O ve NVIDIA performans kabul ölçümleri native Linux aşamasına aittir.

## Hızlı başlangıç

Python 3.11+ gerekir. Doğrulanmış bağımlılık `requirements/runtime.lock` içinde sabitlenmiştir.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements/runtime.lock
make test PYTHON=.venv/bin/python
make demo PYTHON=.venv/bin/python
```

Python, C++ ve açık CUDA-skip kontratını birlikte doğrulamak için CMake 3.24+ ile:

```sh
make check PYTHON=.venv/bin/python
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

## Native doğruluk katmanı

CUDA olmayan bir hostta decoder/oracle ve açık skip testi:

```sh
cmake -S . -B build/native-cpu -DHARPQ_ENABLE_CUDA=OFF -DBUILD_TESTING=ON
cmake --build build/native-cpu --parallel
ctest --test-dir build/native-cpu --output-on-failure
```

RTX 4090 Ubuntu hostunda CUDA 12.0 ile doğrulanan yapılandırma:

```sh
cmake -S . -B build/native-cuda \
  -DHARPQ_ENABLE_CUDA=ON -DBUILD_TESTING=ON \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build/native-cuda --parallel
ctest --test-dir build/native-cuda --output-on-failure
```

`HARPQ_ENABLE_CUDA=ON` toolkit bulunamazsa configure aşamasında, çalışan GPU görünmezse test
aşamasında hata verir. `AUTO` toolkit'i keşfeder; `OFF` CPU testini ve gerekçeli CUDA skip kaydını
çalıştırır. CUDA CTest; materialize-plus-cuBLAS oracle'ını, doğrudan-decode fused GEMV'i ve küçük
pinned-H2D smoke testini ayrı ayrı gösterir.

Gelecekteki `O_DIRECT` ring ile aynı bellek sahipliği modelini kullanan izole H2D denominator'ı:

```sh
build/native-cuda/native/cuda/harpq_h2d_benchmark --json
```

Batch-one doğrudan-decode kernelini aynı launch geometrisine sahip dahili, naive bir static-W4
kernel ile keşif amaçlı karşılaştırmak için:

```sh
build/native-cuda/native/cuda/harpq_gemv_benchmark --json
```

Araç sistem sayfasına hizalı sıradan belleği `cudaHostRegister` ile pinler (ölçülen Ubuntu hostunda
sayfa boyutu 4096 byte), kopyayı CUDA event'leriyle ölçer ve örnek D2H checksum doğrulaması yapar.
H2D çıktısı model throughput iddiası değildir; GEMV karşılaştırmasındaki dahili static-W4 kernel de
üretim/library-parity baseline değildir. Fused GEMV henüz optimize edilmiş static-W4 baseline'a
karşı kabul kapısını geçmiş sayılmaz.

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
