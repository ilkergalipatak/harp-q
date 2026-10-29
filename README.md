# HARP-Q

HARP-Q (Hierarchy-Aware Residual Progressive Quantization), büyük dil modeli ağırlıklarını tek
seferde GPU belleğine yüklemek yerine GPU–RAM–SSD hiyerarşisinde kademeli olarak sunmayı araştıran
bir prototiptir. Ağırlıklar `128 x 128` tile'lara ve sıralı `2+2+2+2` bit residual katmanlarına
ayrılır. Base katman her zaman hesaplanabilir; scheduler zaman ve cache bütçesi elverdikçe residual
katmanları ekler.

NumPy uygulaması formatın ve sayısal davranışın taşınabilir referansıdır. C++17 katmanı aynı stage
payload'larını bağımsız olarak decode eder; CUDA doğruluk oracle'ı seçilen residual prefix'i GPU'da
açıp cuBLAS ile hesaplar. İlk batch-one fused GEMV ise FP32 ağırlık oluşturmadan device-resident
int2 code ve FP16 scale verisini doğrudan tüketir. Native runtime ayrıca hizalı, lease tabanlı bir
packet ring; doğruluk-odaklı buffered `pread`; whole-allocation `cudaHostRegister`; async H2D ve
compute-stream event bağımlılığını içerir. Linux'a özgü CPU doğruluk adaptörü, yalnız
`STATX_DIOALIGN` ile bildirilen geometriyi yetkili kabul ederek `io_uring` ordinary
`IORING_OP_READ` işlemlerini `O_DIRECT` ile doğrudan packet-ring slotlarına gönderir. Bu ilk
adaptör fixed-buffer kaydı değildir. `CudaDirectPackFile` köprüsü, aynı okuyucuyu CUDA kaydı
korunan ring'e bağlar; `CudaPacketTransfer` GPU hedefini ve H2D tamamlanmasını birlikte sahiplenir.
Bu doğruluk yolu henüz NVMe throughput veya kopyalama/hesaplama overlap iddiası taşımaz.
Mac sonuçları yalnızca fonksiyonel doğruluk içindir; NVIDIA ve NVMe performans kabul ölçümleri
native Linux hostunda ayrıca yapılır.

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
cmake -S . -B build/native-cpu \
  -DHARPQ_ENABLE_CUDA=OFF -DHARPQ_ENABLE_IO_URING=OFF -DBUILD_TESTING=ON
cmake --build build/native-cpu --parallel
ctest --test-dir build/native-cpu --output-on-failure
```

RTX 4090 Ubuntu hostunda kullanılan açık CUDA/io_uring doğrulama yapılandırması:

```sh
cmake -S . -B build/native-cuda \
  -DHARPQ_ENABLE_CUDA=ON -DHARPQ_ENABLE_IO_URING=ON -DBUILD_TESTING=ON \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build/native-cuda --parallel
ctest --test-dir build/native-cuda --output-on-failure
```

`HARPQ_ENABLE_CUDA=ON` toolkit bulunamazsa configure aşamasında, çalışan GPU görünmezse test
aşamasında hata verir. `HARPQ_ENABLE_IO_URING=ON`; Linux, liburing ve kullanılan
`io_uring_ring_dontfork`/opcode-probe API'leri yoksa configure aşamasında hata verir. Her iki seçenek
için `AUTO` bağımlılığı keşfeder; `OFF` ilgili test için gerekçeli skip kaydı çalıştırır. CPU CTest
stage codec, packet-ring state machine ve buffered `pread` politikasını ayrı ayrı gösterir.
Dördüncü dilimde eklenen Linux direct-I/O testinin sözleşmesi, canonical padded
pack için `io_uring + O_DIRECT` yolunu buffered `pread` oracle'ıyla fonksiyonel olarak
karşılaştırmak; legacy unpadded terminal tail'i sessizce fallback yapmak yerine açıkça
`buffered-only` olarak sınıflandırmaktır. Beşinci dilim, direct okuyucuyu CUDA registered-ring'e
bağlar; native test sayısı ona çıkar. Ortam ve doğrulama kayıtları `docs/HANDOFF.md` içindedir.
CUDA CTest ayrıca materialize-plus-cuBLAS
oracle'ını, doğrudan-decode fused GEMV'i, registered-ring `pread -> H2D -> stream-wait` zincirini ve
küçük pinned-H2D smoke testini çalıştırır. Yeni köprü testi, direct/pread byte eşleşmesini,
okuyucu/ring yok edildikten sonra lease ömrünü, hata/iptal politikasını ve GPU verisini doğrular.

Hedef NVMe üzerindeki **mevcut** bir test klasörünü seçerek gerçek aktarım kayıtlarını scheduler'a
aktarma ve deterministik replay kontrolü (Python'u `-O` ile çalıştırmayın):

```sh
HARPQ_DIRECT_IO_TEST_DIR=/path/on/target/nvme \
  build/native-cuda/native/cuda/harpq_cuda_direct_packet_io_test --json \
  > build/native-service-observations.json
PYTHONPATH=src python scripts/check_native_feedback.py build/native-service-observations.json
```

Üç boyutta üçer sentetik paket aktarımı yapılır: 64 KiB, 256 KiB, 1 MiB fiziksel okuma; her birinde
29 byte padding vardır. `direct-h2d/host-observed-v1` süreleri CRC, host polling/dispatch ve kritik
yolda paket başına `cudaMalloc` maliyetini içerir. Bu kayıtlar format/replay doğrulamasıdır;
kalibre edilmiş production scheduler maliyeti veya bant genişliği kabul sonucu değildir.

Tekrarlı SSD/H2D ve mevcut C++ runtime karşılaştırması için Release build ile:

```sh
.venv/bin/python scripts/benchmark_transfers.py --build build/native-cuda \
  --scratch /path/on/target/nvme --output benchmarks/results/transfer-sweep
```

Komut yalnız kendi 256 MiB geçici test dosyasını oluşturur/siler; var olan dosyalara yazmaz.
Ön kontrol için `--smoke` eklenebilir. Beş tekrarlı ölçüm yöntemi ve karşılaştırma sınırları
`docs/BENCHMARK_PROTOCOL.md` içinde tanımlıdır.

`O_DIRECT` ring ile aynı host-bellek sahipliği modelini kullanan izole H2D denominator'ı:

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

Scheduler trace v2, tamamlanmış runtime transferlerini sıralı `ServiceObservation` kayıtlarıyla
taşır. Referans politika gözlemleri `(service_class, logical_bytes, physical_io_bytes)` tam
bucket'ında tutar ve yeterli örnek olduğunda kayan pencerenin varsayılan p95 değerini kullanır.
`critical_path_us` deadline/score girdisidir; isteğe bağlı I/O ve H2D bileşen süreleri yalnız audit
metadatasıdır ve overlap uydurmak için toplanmaz. Eski trace v1 kayıtları değişmeden okunup replay
edilir.

Araştırma kapsamı ve iddialar için `docs/RESEARCH_PLAN.md`, önceki çalışmalar için
`docs/PRIOR_ART.md`, mimari kararlar için `docs/DECISIONS.md`, makine geçişi için
`docs/HANDOFF.md` dosyalarına bakın.
