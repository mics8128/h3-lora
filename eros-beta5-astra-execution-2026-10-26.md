# Astra 長序列最佳化諮詢：執行調度與記憶體生命週期

> 本文保存 `astra_consult` 明確路由 `llm-esapp/gpt-6-astra` 的實際諮詢結果，非協調代理自行替代分析。諮詢已提供完整共用契約與 execution/memory 專項要求。**本次未執行 inference、benchmark、測試或 build；未生成影片、修改 runtime、安裝資源或停止服務。** 生成維持暫停。父代理已選 4 steps，離線準備通過的 2 actors／5 images／2 audios／141 frames 不是新的推理測試；下列既有計時來自單圖單音訊樣本，不可外推為該工作的新量測。

## 範圍與核心判斷

分析基準：768×1344、24fps、50 layers、reuse/core=1、4-step ER-SDE beta57、shifts 12/3；INT8 projections、BF16 video VAE、F32 audio VAE。保留 beta5 rank8/time-table、BF16 RMSNorm→F32 final modulation/heads、joint A/V attention/RoPE 與 reference isolation。40GiB 為影片程序目標，並不代表 LLM 可以免費同時重 GPU 推理。

來源識別採契約提供 snapshot `0f96a9bd859f1576097327e2b7f46298b28d31d8`、binary `e5a88771f646cfc4bcc01b87011255ae10a14c78c47055b92fd37af2ba8689ab`；本次未重算 hash，不宣稱新驗證 source/binary 一致。

1. **412.794s wait 不是可刪除的同步 overhead**：包含實際 GPU/MPSGraph 完成時間；root timestamp 不涵蓋所有 child buffers。
2. ER-SDE 確實每步往返 CPU，但不是每層往返，也不是 PCIe transfer：shared Metal buffer 的 memcpy、packing、BF16 widening、CPU sampler 與 producer/consumer barrier。
3. activation arena 已跨層跨 step 重用；denoise `alloc=0.000GiB`，再提「不要每步重建大 tensor」不對症。
4. persistent context 改善連續工作啟動成本，不會把 489s 長序列變成數十秒。
5. 顯著改善相同四步 workload，必須改善 200 次 block execution 的運算或 framework scheduling；immutable cache／sampler 搬移沒有支持大幅加速的證據。

## 直接來源追蹤

### 應用每 job 新建 native process

`h3.py`：`prepare_video()` → `render_key()`/guard → immutable inputs → SCP request/runner → SSH；`h3_remote.py::main()` 取得 engine lock、檢查其他 h3、`subprocess.Popen(h3 -p ...)`、`child.wait()`。Python wait 是 job 控制，不是每 step GPU 同步；改 asyncio 不消除 DiT 計算。

應用來源：[h3.py:334–410](https://github.com/mics8128/video-maker/blob/main/src/video_maker/h3.py#L334-L410)、[h3_remote.py:85–100](https://github.com/mics8128/video-maker/blob/main/src/video_maker/h3_remote.py#L85-L100)、[138–177](https://github.com/mics8128/video-maker/blob/main/src/video_maker/h3_remote.py#L138-L177)。

### ER-SDE 與 GPU Euler 是不同路徑

`h3.c::h3_generate()` 非 serving scheduler → `h3_dit_denoise_simple_preview()` → `h3_sampler_run()` → `h3_sampler_state_run()` → `simple_evaluate()` → `h3_dit_forward()` → CPU clean prediction/ER-SDE/history update。

既有 `denoise_euler_gpu()` 有 GPU latent state 與 bounded step window，但 **H3_GPU_SAMPLER_WINDOW 不會替目前 ER-SDE 開啟 GPU-state sampling**。

來源：`h3.c:2194–2204`；`h3_dit.c:2918–2990、2719–2730、2743–2915`。

### 每個 ER-SDE step

1. CPU video patchify/audio pack。
2. 寫入 video/audio input 的 target range；reference rows 只在 prepare/reset 寫。
3. `encode_forward(begin=1,submit=1)` 執行 50 blocks、final modulation、F32 heads。
4. output cast BF16，`h3_gpu_submit()` 等待。
5. CPU 讀 BF16 velocity→F32、unpatchify/unpack。
6. CPU 更新 latent、prediction/derivative histories、RNG。

`h3_gpu_tensor_new()` 使用 `MTLResourceStorageModeShared`；read/write 為 `buffer.contents` memcpy。消除 transfer 指少 memory traversal、packing、完成屏障，不是省掉獨立 GPU PCIe DMA。

來源：`h3_dit.c:2528–2637`；`h3_gpu.m:463–493、728–787`。

### 現有 scheduling 已部分 overlap

M5 預設 block interval 為 `50×3/5=30`。`encode_forward()` 在第30 block 後 `h3_gpu_continue()` commit 不等待、同 ordered queue 繼續；step 最後 submit 才等整條 chain。四步日誌 8 submissions、200 attention dispatches 與此一致；不是50層各自 wait。

SSD streaming 分支每層 submit/join 下一層 I/O，是另一省記憶體路徑，不能當現行 INT8 resident 瓶頸。

來源：`h3_dit.c:213–224、2283–2408`；`h3_gpu.m:789–863`。

## 既有數據與可推論上限

| 四步既有指標 | 數值 |
|---|---:|
| 端到端 | 489.602s |
| DiT load | 13.635s |
| denoise wall | 413.015s |
| encode | 0.022s |
| wait | 412.794s |
| root GPU | 2.718s |
| submissions / attention dispatches | 8 / 200 |
| denoise allocator 新配置 | 0.000GiB |
| tracked DiT peak | 25.524GiB |
| video VAE | 53.973s |
| sampled physical footprint | 26.309GiB |

來源：[fast-four/stderr.log:13–22](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/outputs/perf-20261026-memory40-long/fast-four/stderr.log#L13-L22)。

`h3_gpu.h:17–33` 明說 root command timestamps 不涵蓋所有 MPSGraph child work；`h3_gpu_submit()` 只加 root timestamps。因此 **不能用 413.015−2.718 宣稱有410s CPU/driver 空轉**，也不能據此算 attention 比例。

**[條件式計算]** 若 GPU 工作與 wait 不變，即使消除全部 wait 外 wall，最多省 `413.015−412.794=0.221s`，約端到端0.045%。GPU-state 可另改 scheduling/page behavior，但目前不支持 CPU sampler/memcpy 已吃數十秒。

六步 denoise653.864s、四步413.015s，簡單平均每eval108.98/103.25s；VAE也不同，非獨占、filesystem cache未清。不是因果 benchmark，不能外推雙人多圖。

## Activation lifetime 與 40GiB

`allocate_activations()` 一次配置跨 step：BF16 hidden/mod_attention/QKV、qkv arena、attention output、INT8 activation/scales、MLP、F32 final target-row scratch、latent I/O。

預設 alias：attention_heads 與 mod_mlp 在不同生命期共用 qkv arena；MLP output 共用 attention_output。`run_block()` 把 MLP gate 與下一 block AdaLN/INT8 quantization 融合；切 window 必須保留跨 block producer/consumer。

來源：`h3_dit.c:1369–1567、2071–2148`。

令 joint packed tokens=N：HIDDEN5376、INNER7168、FFN14336：

| 配置 | bytes |
|---|---:|
| BF16 hidden | 10,752N |
| Q/K/V 合計 | 43,008N |
| qkv alias arena | 43,008N |
| BF16 MLP activated | 28,672N |
| INT8 activation（未含padding/scales） | 14,336N |
| F32 final scratch | 21,504N_target |

**[非樣本實測，量級例]** N=45,000：hidden0.451GiB、QKV1.802GiB、另一qkv arena1.802GiB、MLP1.202GiB、INT8 activation0.601GiB。

令目標latent元素總數L：ER runtime storage是六份F32 arrays=`24L` bytes；mask另加三份。checkpoint七份serialized arrays不是runtime必常駐七份（`h3_sampler.c:68–96、345–350`）。

26.309→40GiB的13.691GiB差值不是安全新增scratch：抽樣可能漏峰、tracked bytes未含framework、常駐改變phase overlap、mapped/filesystem/footprint不能混帳。`h3_generate()` 非cached DiT在VAE前釋放，cached DiT則保留到decoder；RGB F32與U8短暫共存。**[計算]** 141×768×1344×3×(4+1)約2.03GiB，尚未含decoder intermediate。

來源：`h3.c:2217–2273`；`h3_dit.c:3159–3212`。

## 排名一：有界長序列 GPU executor，再依證據替換 SDPA

**既有** ordered queue、block split、MPSGraph SDPA；**需開發** bounded retirement、workspace quota、條件式 tiled SDPA adapter。

在 `encode_forward()`、`h3_gpu_continue/submit()` seam 統一 ordered submission、completion retirement、resource lifetime、workspace budget。

A. 保留 kernels，後續比較30/10/5-block windows；commit與wait分離，每window completion ticket，in-flight超界只回收最舊窗口。初始最多2 windows只是設計起點。維持單ordered queue、原arena alias，不複製完整activation。現有continue累積inflight直到submit才清，不是真固定容量逐項retire。

B. **只有完整GPU timeline證實SDPA為主要熱點**才在 `h3_gpu_sdpa()` 新增query/KV tiling＋online softmax；保留完整joint A/V/text/reference、RoPE/layout/scale、全K/V可見。不是local/sparse attention。exact表示不做模型近似，不保證浮點bitwise。

MPSGraph內部NxN未知；N45k、56heads的完整BF16 score約211GiB是量級計算，不能證明現況真的配置它。

40GiB新增預算優先有限tile workspace/in-flight scratch，不是重存所有BF16+INT8 weights或雙contexts。`h3_gpu_tensor_free()`會設purgeable empty；完成前不能free，光保留ObjC reference不夠。

**最小判別實驗（未執行）**：固定長序列seed/kernels只改window，完整捕捉MPS child timeline；若只增加submissions無穩定收益，回30。scheduling要求每步velocity/final latent bitwise；tiled attention先isolated oracle，再四步誤差與語義。

**[條件式上限]** attention占denoise比例f、加速r：`T'=489.602−413.015f(1−1/r)`。若attention2倍，f20%→約1.09倍端到端；f50%→約1.27倍。f未知，不是預測。

成本高；fallback現有kernels/window；不得改frames/resolution/references/50layers/NFE。

來源：`h3_dit.c:2283–2408`；`h3_gpu.m:713–726、789–863、1391–1516`。

## 排名二：GPU-state ER-SDE，保留 CPU oracle

**既有** GPU Euler、CPU ER；**需開發** GPU state/history kernels、RNG/commit/checkpoint一致性。主要價值是可跨步調度seam，不是已有大額CPU瓶頸。

Interface：`prepare(config, initial_state)`、`encode_transition(step,velocity)`、`materialize_committed_state()`；solver coefficients/sigma與checkpoint仍由sampler擁有，不拆A/V獨立solver。

必保留video/audio shift ratio、carried/native mapping、velocity sign、history更新順序、first-sigma前initial noising、terminal next=0不取noise、mask只投影完成proposal、失敗不提交RNG/history/step。F32heads後的BF16 velocity rounding不能省。

第一版CPU產生原Gaussian與coefficients，GPU執行當前DiT時CPU預算下一非終點noise；雙noise buffer ownership、speculative與committed RNG分離、native→patch index正確。原CPU double log/sqrt/sin/cos移GPU不保證同noise，故不一開始全搬。

**Gate** 先每transition完成點，再跨兩步window；同velocity/noise的state/history/RNG bitwise。FMA/順序差異先修正，不偷稱exact。checkpoint維持solver3/revision1 host schema；pause/resume/softmask/startstep/terminal-zero語義不變。不支持配置在開始前回CPU；中途失敗不得未知狀態重送。

**效益** 直接可消除wall目前小；可少host materialization、為跨步scheduling鋪路，不能說「消掉4次wait就省413s」。工程成本高，若profiling仍證明step boundary微小則不當速度優先項。

來源：`h3_sampler.c:198–340`；`h3.c:1270–1275`；`h3_dit.c:2480–2513、2600–2634`。

## 排名三：單 owner persistent context＋weights-only tier

**既有** native cache/reset/reprepare；**需開發**正式job協定、ownership、取消/unknown-result語義與memory lifecycle。不要用互動CLI文字stdin當正式協定。

以h3_ctx/h3_generate做單owner序列worker：相同conditioning/shape走reset，相容weights不同prompt/layout走prepare_request，相同sigma/modality保留AdaLN schedule，decoder按key重用。

三級residency：prepared（weights/schedule/request arena/相容decoder）；weights-only（釋放request arena/host staging，必要淘汰decoder）；unloaded。`clear_request()`已有內部釋放能力，但獨立tier及prepared-key重設仍需實作。

有相容next job可prepared；無job或LLM需求轉weights-only；不足則unload自己的worker，不停其他服務。TTL不取代memory-pressure判斷。

Reference key目前含prompt/reference順序/path/file identity；job新path即使內容同仍可miss。第一階段接受conservative miss；若改content identity，必保role/order/audio attribution/model/adapter/precision/shader/shape/frame/preprocess。seed-dependent augmentation不得污染immutable cache；現有先cache再augment順序保留。

Remote現有拒絕其他h3、以child PID/group取消；常駐需分worker與job ownership、唯一jobID、disconnect只取消所屬job、unknown result查同job不重送、保留render_key/input hash/diagnostics/媒體發布驗證。

### Cache有限上限

**[計算]** 非denoise76.587s：即使全部消失（連VAE都不合理算掉），最多1.185倍／15.64% wall reduction。

保留VAE53.973s，餘22.614s即使歸零也僅1.048倍／4.62% reduction；這是VAE stage不變的條件包絡，不是嚴格cache上限，decoder load可能在VAE內。

可辨DiT load、Qwen text/vision、video/audio reference encoders合計20.737s，即全省約4.24% wall／1.044倍；實際仍有reprepare、新prompt encoder與key cost。

**Gate** fresh/warm相同seed latent相同；A-reference→B→A第三次與fresh A同；長短序列交替retained memory可回tier；warm超標降weights-only，再超則一次性job。與新kernels分開rollout。

來源：`h3.c:285–317、1973–2036`；`h3_dit.c:1650–1687`。

## 排名四：明確 opt-in 單一中間 step residual reuse（近似）

**不是fast4steps等價模式，不是作者推薦，不是現成開關。** 保留4個sigma transitions，第0/1/3 step full，只在第2 step研究重用step1中間blocks1–48 residual，首尾blocks/heads重算。需記錄每step fresh/reused blocks，不得只記layers50誤稱完整運算。

現有非serving強制full layers/reuse=1；core cache是whole-core residual，不能簡單改core-reuse=2。ER history會累積誤差；sigma modulation變；joint attention令reference hidden依target改變，「reference不變」不代表跨步K/V exact cache。作者reference/Spectrum accuracy警告仍適用。

**[均勻block成本樂觀計算]** 省48/200=24% block work；若整個denoise都等價，可省至多99.12s，端到端約390.48s／1.25倍；實際copy/gate/heads使收益更低，品質可能完全不合格。

額外residual/input snapshot約兩份N×5376 BF16，N45k示例0.90GiB。僅research profile；gate必含多圖雙人、身分、speaker/voice attribution、動作、嘴型/聲畫同步；不能只用平均SSIM/latent error。identity drift/reference串用/台詞歸屬退化即回full四步，不偷偷加steps後宣稱同方案。

來源：`h3.c:790–799`；`h3_dit.c:2268–2282、2414–2423`；`EROS_BETA5_SETUP.md:37–39`。

## 共同 gates、LLM coexistence 與 rollout

數值契約來源：`h3_dit_schedule.c:78–91、118–175` 的1025×8 clamp/linear curve、t=1−sigma及curve特殊路徑；`h3_shaders.metal:3082–3121` BF16round→F32scale/add且無乘加contraction；`h3_dit.c:2425–2513` F32heads後BF16velocity。

- scheduling/persistent/lifetime：每step velocity及final latent bitwise。
- GPU ER：固定velocity/noise下state/history/RNG bitwise。
- tiled SDPA：先baseline repeatability與事前固定max-error/relative-L2，再四步語義；relative-L2≤1e−3僅isolated研究起始門檻，非已證明beta5品質界線，不得失敗後臨時放寬。
- residual reuse：明確近似品質gate，不能宣稱exact。

### LLM：memory admission與compute admission分離

Memory必估 `video peak + LLM weights/KV/scratch + OS/other + margin`；128GiB與video40GiB不代表其餘皆可用。idle resident LLM仍有memory成本。

Compute即使memory足夠，LLM prefill/decode仍爭GPU算力/頻寬/cache/功耗。bounded windows不保證跨process公平。重GPU預設錯開，LLM可resident但同時重推理需獨立compute admission。現`.video-maker.lock`/ps只涵部分H3，不是跨LLM全域scheduler。

Memory policy建議40GiB正常目標；小幅overshoot需明訂如研究暫定42GiB，非無限寬容。同看process footprint、Metal/framework、system pressure/swap。超標先減window scratch、清request、淘汰decoder，不降規格。抽樣未超不等於所有reference/長度安全。

### 推進順序（未執行）

1. 固定四步長序列取得完整MPS child timeline，不再從root猜瓶頸。
2. 只改window，同kernels/seed/precision；無穩定收益即回預設。
3. SDPA占比高才做tiled；projection高不可錯怪attention；CPU boundary小則GPU ER不優先。
4. 獨立驗收persistent，避免kernel數值/cache isolation/memory峰值混淆。
5. exact路徑有證據後才近似，獨立profile/產物標記，不自動替換fast4steps。

未來性能推廣需交錯同條件、cold/warm分開、固定其他GPU工作，改善超過run-to-run波動。本次未做上述實驗，**所有方案均未獲新效能或品質核准**。

## 原始來源索引

遠端來源根：`ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/`。

- `h3.c`：285–400、790–799、1259–1275、1430–1501、1973–2036、2190–2273。
- `h3_dit.c`：16–204、213–224、728–776、1369–1687、1979–2148、2153–2637、2711–3022、3159–3238。
- `h3_gpu.m`：137–166、463–589、713–863、1391–1516、2384–2419。2048/3072是projection policy，不是SDPA切換。
- `h3_gpu.h`：17–98。
- `h3_sampler.c`：12–27、68–96、118–124、198–400。
- `h3_dit_schedule.c`：78–175；`h3_shaders.metal`：3082–3121。
- `RUNTIME_REFERENCE.md`：115–141；`EROS_BETA5_SETUP.md`：37–39、83–87。
- 既有日誌：`outputs/perf-20261026-memory40-long/fast-four/stderr.log:3–22`、`fast-six/stderr.log:13–23`。
- 本地背景：[eros-beta5-performance-options-2026-10-26.md](eros-beta5-performance-options-2026-10-26.md)。
