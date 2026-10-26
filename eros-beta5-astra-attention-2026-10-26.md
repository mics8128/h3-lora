# Eros Max beta5：Astra attention 演算法設計研究

日期：2026-10-26。**本文件由實際 `astra_consult`（指定 llm-esapp/gpt-6-astra）諮詢結果整理，非以協調模型替代 Astra 分析。僅唯讀檢視來源；未執行推論、生成、測試、編譯、benchmark 或安裝。**

使用者已選 fast4steps；主代理另回報雙角色／5 圖／2 音訊／141 幀已完成離線準備並具有獨立 render key，生成仍暫停。以下既有單圖單音訊測量不是該多參考工作的效能證據。Astra 未對新工作執行 tokenizer、encoder 或推論，故不虛構其精確總序列長度。

## 1. 結論與優先順序

優先研究完整 joint attention 的 **Metal tiled／online-softmax**；保留 MPSGraph 作基準與回退。Exact 指保留所有 attention 邊及同一 softmax，不保證不同浮點歸約逐位一致。不得先把 sparse 當 beta5 預設。

| 排名 | 具體候選 | 語義／現況 | 工程及主要風險 | 最小判別實驗 |
|---|---|---|---|---|
| 1 | 原生 Metal tiled online-softmax | 完整 dense；尚須實作 | 高；PV 精度、register spill、framework 可能已很快 | 正式 56×N×128，32×64、32×128 tiles，與同份 QKV 的 MPSGraph 比較 |
| 2 | MPSGraph rectangular query 分塊、完整 K/V | 完整 dense；須新增 adapter | 中低；額外 dispatch／gather，framework workspace 未知 | Bq=512/1024；完整 graph 對照，計入資料搬運 |
| 3 | Split-KV online-softmax＋精確合併 | 完整 dense；依賴候選1 | 中高；partial-output scratch 與 reduction 流量 | 只有候選1出現長迴圈／occupancy 問題才比較 S=1/2/4 |
| 4 | AV/reference-safe local／sparse／dense hybrid | **近似、永久明確 opt-in** | 品質工程最高；身份、動作、嘴型風險 | 只裁 target-video↔target-video 邊，完整 AV/reference gates |

四者都不是現成可用開關。候選2可作較小的早期實驗。Exact tiled attention 減少 IO／中間儲存，並不消除 O(N²) 算術。

## 2. 已安裝契約及來源界線

原生來源：`/Users/tedmacbookpro/projects/h3-lora-eros-beta5`（ssh://ted）；既有 provenance 記錄 snapshot `0f96a9bd859f1576097327e2b7f46298b28d31d8`、binary SHA256 `e5a88771f646cfc4bcc01b87011255ae10a14c78c47055b92fd37af2ba8689ab`；此次未重新雜湊。

M5 Max 128 GiB；full BF16 beta5 TURBO-hybrid 來源，當前 INT8 projections、BF16 video VAE、F32 audio VAE。固定 768×1344、24fps、50 layers；ER-SDE/beta57、4 steps、shift12/3、reuse/core-reuse1。保留 rank8／[1025,8] time-table、BF16 RMSNorm→F32 final modulation／heads。影片程序目標約40GiB，留 resident LLM／OS 餘裕，不假設重GPU工作可免費並行。不得暗減解析度、片長、reference、層數。

`h3_dit_curve_interpolate()` 保持 [1025,8] clamp/floor/linear interpolation；curve final 使用 `h3_gpu_adaln_bf16_f32_offset()` 及 F32 heads，attention 改造不觸及。

## 3. Packed production sequence：56 heads、D=128

`h3_dit.c:16–30`：hidden=5376，heads=56，head_dim=128，attention inner=7168。**不能以5376/128推得42 heads。** Batch=1；INT8投影輸出 BF16 Q/K/V，主路徑 head-major `[1,56,N,128]`，scale=1/√128，非causal audiovisual joint self-attention。

### Target 141 frames

`h3_host.c` 將 target frame 對齊至 5+17k；target latent time 為 `5*(F-5)/17+2`。141幀得42；VAE spatial ratio16，latent 48×84；2×2 patch後每frame 24×42=1008 tokens。

- Target video：42×1008＝**42,336**。
- Audio latent time：round(141×40/24)＝235。
- 兩 audio streams，每stream235 rows（不是batch），共**470**。
- Target audiovisual 小計：**42,806**。

一般式：

`N = T_text + 42,806 + R_image + R_video + R_audio + G_video + G_audio`。

- Image reference：`(latentH/2)*(latentW/2)`。
- Video reference：`encodedT*(latentH/2)*(latentW/2)`；其encoder時間是 floor((F+3)/4)，不是target時間公式。
- Audio reference：`2*encodedAudioT`。
- Guides 另外加入，不能因與target描述同時段就扣除target rows。

實際順序：text → guides各video/audio segments → references依輸入順序（同一video+audio reference內audio先於video）→ target audio stream0/1 → target video time/spatial rows。`validate_layout()` 檢查contiguous segments與payload；`dit->sequence=layout->seq_len`。

新雙角色5圖2音訊工作應使用：

`N_new = 42,806 + T_text + Σ(i=1..5) imageTokens_i + 2*Σ(j=1..2) audioLatentT_j`（若無額外guides/video refs）。

此為依實際layout公式的計算式，不是已觀測的總N。每張圖的`match`保持比例、只縮小並對齊，不保證都是768×1344；雙角色不另乘batch或heads。既有單圖單音訊樣本亦只能確認 `N=42,806+T_text+R_person+2*T_refaudio`；若圖canvas恰為768×1344才有R_person=1008。Astra嘗試唯讀PNG但ssh工具拒絕binary；既有log/result未記錄text/reference latent rows，故未補造總數。下一次獲准實驗應直接記錄layout spans與seq_len。

### 算術與記憶體尺度

Dense QK+PV約 `4*56*N²*128 = 28,672*N² FLOPs`。以**示例N=45,056（不是樣本實測）**：

- 每份BF16 Q/K/V/O：616MiB；四份2.40625GiB。
- 若全部heads完整score materialize：BF16 211.75GiB／F32 423.5GiB。
- 每層QK+PV約58.2TFLOP；50層×4 evaluations約11.6PFLOP；皆非時間預測。

現有activation另有三倍inner的`qkv` arena及獨立Q/K/V，`attention_heads`／`mod_mlp` alias qkv。示例六份inner BF16約3.609GiB，尚不含hidden/FFN/weights/framework。不可因fused QKV就直接刪qkv，其後仍作scratch。[INFERENCE] 26.3GiB既有footprint不支持所有heads完整score同時常駐，但不能排除逐head／分塊／暫態workspace。

## 4. 現行路徑與framework未知

`run_block()`：AdaLN→INT8 QKV→BF16 Q/K/V→QK RMSNorm與learned weights/RoPE→MPSGraph BF16 noncausal SDPA→head-major INT8 output projection→residual/MLP。

`h3_gpu_grouped_qkv_linear_rope_int8()` 設headMajorSDPAInputs；`h3_gpu_sdpa()` 消費並清除；graph可不新增QKV transpose，下游`h3_gpu_linear_int8_head_major_bf16()`直接接head-major output。**消除所有transpose不是尚未存在的新優化**；graph transpose也不代表一定materialize。

`h3_gqa_causal_bf16()` 是另一條text causal GQA，保存每query key scores，先scale Q並round BF16；不是56-head DiT FlashAttention，不能直接套用其數值語義。

未知：MPSGraph fused/tiling、累加與softmax精度、scale位置、score/probability workspace、transpose是否copy、child command buffers完整時間。2048/3072是部分BF16 projection shape門檻；INT8 QKV rows>2048影響local-scales kernel，**不是SDPA backend switch**。

既有4step log：denoise413.015s、wait412.794s、root-gpu2.718s、attention200 dispatches。200與50×4一致，但無逐項attention latency。submit wait包含GPU執行；header明載MPSGraph可能child command buffers。不能說410秒是同步浪費或GPU只工作2.718秒。[INFERENCE] N²值得調查，但projection/FFN亦巨大，不能由FLOPs比例推出耗時。

## 5. 候選1：native online-softmax具體設計

每threadgroup處理一head的Bq個query，掃描所有K/V tiles。每row維護F32 running max m、normalizer l、128維未正規化u，初值(-∞,0,0)。每tile：

```
S = Q K_tile^T / sqrt(128)
m_new = max(m, rowmax(S))
a = exp(m-m_new)
P = exp(S-m_new)
l_new = a*l + rowsum(P)
u_new = a*u + P*V_tile
O = BF16(u/l)
```

完整joint softmax，不寫全域N²；QKV/O維持BF16，m/l/u F32。不跨步重用。

Seam：`h3_gpu.m:h3_gpu_sdpa()`新增adapter；第一版只接受batch1/BF16/noncausal/56heads/D128 DiT；其他shape維持MPSGraph。`h3_shaders.metal`新增kernel；保留run_block上下游head-major契約。

先比較Bq×Bk=32×64、32×128。32×64若全列出：Q8KiB、K16KiB、V16KiB、F32 scores8KiB、F32 output accumulator16KiB；**不可全部塞threadgroup memory**。Q/fragments/output盡量register/cooperative，K/V分階段重用staging。受裝置／pipeline實際限制；spill可能使大tile更慢。

尾端keys以-∞屏蔽、queries不越界讀寫；不讀padding V；空tile避免-inf-(-inf) NaN；每有效key恰使用一次。

M5可行基礎：現有Metal有mpp::tensor_ops、matmul2d、native bfloat及optional compile fallback；Apple有TensorOps FlashAttention官方示例。但已安裝SDK對新cooperative tensor作下一matmul輸入、reduce_rows等未確認；WWDC26指出macOS26需threadgroup staging，不能當新SDK能力已存在。

第一版：QK matmul2d用F32 accumulator、自有SIMD max/sum、PV保留F32 P語義；mixed dtype不可用時tile內widen V或F32 SIMD correctness路徑。**不能偷偷把P轉BF16換速度**；若做，另列precision variant並單獨驗收。不一起融合projection、改RoPE rounding或fast-math。

Capability不符在dispatch前fallback；head-major旗標只消費一次。GPU執行失敗明確報錯，不在未知狀態靜默重送生成。

## 6. 候選2：rectangular MPSGraph

Q=[1,56,Bq,128]、K/V=[1,56,N,128]，Bq=512/1024；每query仍看全部keys。現有graph只有sequence，須拆query_length/kv_length、shape/cache key、input/output subrange。

Head-major `[H,N,D]` 的query block跨heads有N*D stride，不能只加buffer offset假裝contiguous `[H,Bq,D]`。用graph slice/stride或bounded gather/scatter，計入copy；不可每塊複製完整KV。不可切KV後平均output或按modality各自softmax再加總。

[INFERENCE] 矩形shape可能改變framework kernel/workspace，也可能因dispatch及重讀而退步；維持O(N²)，只在正式測量勝出時保留。

## 7. 候選3：split-KV精確合併

每partition產生F32(m_s,l_s,u_s)：

```
m=max_s(m_s)
l=sum_s exp(m_s-m)*l_s
u=sum_s exp(m_s-m)*u_s
O=u/l
```

Partitions完整、不重疊地覆蓋keys。N45,056/Bq32已有78,848 query threadgroups，不能先認定平行度不足。Split增加Q重讀、scratch、reduction；S4全query F32 partial u/m/l約4.89GiB。可query windows限制但增加dispatch。只有候選1 profiler支持時比較S1/2/4，無穩定收益即刪除變體。

## 8. 候選4：明確近似的AV/reference-safe hybrid

令V=target-video tokens，G=其他全部tokens（text、target audio、references、guides）。保留G queries→全G/V、V queries→全G；僅稀疏V→V。首版V local取同latent frame全部spatial、原model-time鄰近frames，選配同spatial跨時間；集合聯集去重，共享一個softmax denominator。

若M=|V|、每video query平均w個video keys，pair count約G²+2GM+Mw，非wall-clock倍率。

保守實驗schedule：evaluation1/4全dense；evaluation2/3只在中段layers sparse，前後及定期global layers dense。仍50layers/4eval/56heads；dense refresh不保證恢復失去的身份/動作。

### AV／reference／RoPE gates

- 原positions不等於packed row距離：text自己的index、image reference cursor+1、audio依latent time、video依AV span；target video時間步循環(1,4,4,4,4)×5/3。
- 兩audio streams共享時間但spatial不同；t/h/w各16 frequencies共48pairs，旋轉128維前96維。
- local鄰近依原position/segment，不按packed index；不重編reference時間、spatial scaling、不合併不同素材tokens。
- 保留target video↔target audio雙向全連接，以及所有reference/text/guides原可見性。
- 現有dense沒有reference彼此隔絕mask；isolation指素材身份/payload/positions/request ownership，不能新增block-diagonal mask假冒原語義。

固定reference輸入不代表後續每層reference KV固定：joint attention令其hidden接收target資訊，target跨denoise變化。跨步reference KV重用是近似，不能與精確graph/RoPE/前處理cache混同。作者reference cache/Spectrum accuracy warning不等於批准sparse；beta6動態README建議不能外推beta5。

## 9. 尚未執行的最小實驗與驗收

**所有以下項目須另獲准；本次未執行。**

1. **E0形狀／歸因**：記seq_len與segment spans/reference dimensions、head-major flags；完整GPU capture含MPSGraph children，分SDPA/QKV/output/FFN、framework workspace、footprint與system pressure。自有peak_live_bytes不是全部framework allocation。若SDPA非主成本，停止擴張attention工程。
2. **E1 exact比較**：固定同一QKV，先小型oracle（D128、整除/尾端、均勻/尖峰/重複K/極端幅值），再正式56×實際N；分首次compile與steady-state；比較SDPA單項及QKV→SDPA→output短鏈，計入搬運。不要用2048token替正式形狀。
3. **數值gate**：dense keys恰使用一次，layout/RoPE/scale不變；finite、無越界/alias hazard；對獨立高精度oracle。建議起始BF16 relative RMS≤5e-3，近零另設absolute gate，按head/modality與尾部分布檢查。這是預先建議門檻，未通過、非普適品質保證；不得結果出爐後才放寬。
4. **E2 split**：只有E1證據需要才S1/2/4，量spill/occupancy/bandwidth/scratch及約40GiB目標；重複交錯順序，單次差距不算收益。
5. **E3媒體語義**：固定seed/initial latent/4step/projection，先單層/單eval再最終影音。覆蓋多reference雙人身份綁定、聲音/台詞/嘴型/說話者、大位移轉身遮擋再出現、長時背景、request換素材隔離；單張揮手樣本不夠。Exact亦不能免媒體驗收。近似模式對身份漂移、錯誤說話者、少字加字、嘴型錯位設否決，失敗回全dense，不減reference/片長/層數。

## 10. 上限、記憶體與推出

設attention占比f、其加速s，端到端上限1/((1-f)+f/s)；f尚未量，不能承諾倍率。既有489.602s中denoise413.015s；**即使整段denoise歸零**也剩76.587s，算術寬鬆上界6.39×，不是attention預測。

Native維持QKV/O及alias lifetime，不新增全序列F32 copies；rectangular限定query scratch；split明列partial scratch；sparse用tile descriptors不建N²mask。26.309GiB單樣本不保證新5圖2音訊仍同峰值。

推出：MPSGraph保持預設；native先opt-in限定已驗收shape/dtype/layout；數值、正式形狀效能、記憶體與影音reference品質全過才考慮exact預設化。Sparse維持獨立近似模式。不把Spectrum、reuse、裁層、降解析度合併驗收。不停止現有服務；重GPU工作分開排程。

## 來源與直接檢視定位

以下是Astra此次直接檢視的安裝來源定位；非本協調代理再次執行驗證。

- [S1 h3_dit.c:16–30、1980–2059](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_dit.c:16-30)：dimensions、run_block QKV/SDPA/output。
- [S2 h3_host.h:7–11](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_host.h:7-11)、[h3_host.c:12–44](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_host.c:12-44)、[74–97](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_host.c:74-97)、[299–387](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_host.c:299-387)：frame alignment、latent geometry、image canvas、grids。
- [S3 h3_host.c:395–625](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_host.c:395-625)：h3_layout_build、segments/positions。
- [S4 h3_dit.c:297–410](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_dit.c:297-410)：validate_layout；[h3.c:1550–1569](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3.c:1550-1569)、1619–1630 image size；1700–1737 audio；1994–2002 layout spec。
- [S5 h3_gpu.m:1389–1553](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_gpu.m:1389-1553)：h3_gpu_sdpa_graph/h3_gpu_sdpa；[h3_gpu.h:507–517](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_gpu.h:507-517)。
- [S6 h3_gpu.m:340–405](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_gpu.m:340-405)：M5 optional compile/fallback；[h3_shaders.metal:929–955](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_shaders.metal:929-955)、[1956–2110](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_shaders.metal:1956-2110)：matmul2d、INT8 QKV head-major/QK norm/RoPE。
- [S7 h3_gpu.m:3868–4004](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_gpu.m:3868-4004)：h3_gpu_grouped_qkv_linear_rope_int8。
- [S8 h3_dit.c:1369–1455](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_dit.c:1369-1455)：allocate_activations與aliases。
- [S9 h3_gpu.h:17–34](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_gpu.h:17-34)；[h3_gpu.m:184–212](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_gpu.m:184-212)、480–488、827–860：profile、accounting、submit。
- [S10 runtime-source.json](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/runtime-source.json)；既有[command](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/outputs/perf-20261026-memory40-long/fast-four/command.json)、[stderr](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/outputs/perf-20261026-memory40-long/fast-four/stderr.log)、[result](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/outputs/perf-20261026-memory40-long/fast-four/result.json)。
- [S11 h3_dit_schedule.c:78–91](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_dit_schedule.c:78-91)；[h3_dit.c:2429–2439](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_dit.c:2429-2439)、2491–2512：rank8/time-table與F32 final。
- [S12 h3_shaders.metal:4013–4085](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_shaders.metal:4013-4085)：causal GQA不同路徑。
- [S13 h3_gpu.m:2384–2419](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_gpu.m:2384-2419)：BF16 projection門檻。
- [S14 FlashAttention primary paper](https://arxiv.org/abs/2205.14135)：IO-aware exact attention；其他硬體benchmark不外推本機。
- [S15 Apple WWDC26 Metal TensorOps](https://developer.apple.com/videos/play/wwdc2026/330/)：9:31起cooperative tensors、row reductions、FlashAttention與macOS26 staging限制。
- [S16 作者動態README](https://huggingface.co/TenStrip/10Eros-Max/blob/main/README.md)：beta5 sampler及reference cache/Spectrum warning；不是固定版本驗收。
- [S17 h3_dit.c:899–942](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_dit.c:899-942)、974–994：prepare_rope、原positions及BF16 table。
