# Astra 獨立設計諮詢：長序列 INT8、kernel fusion 與權重預封裝

> 本文保存 `astra_consult` 明確路由至 `llm-esapp/gpt-6-astra` 的實質回覆；由協調代理整理保存，未以自身模型替代諮詢。Astra 僅唯讀檢查來源與既有紀錄，未執行推論、影片生成、benchmark、測試、建置、安裝、下載、遠端修改或服務操作。下列實驗均未執行。
>
> 工作狀態補記（Main 提供，非本次驗證）：已選四步；雙角色／五張圖片／兩段音訊／141 frames 的 offline prepare 已通過，具有 distinct render key；生成維持暫停。本次既有單圖效能樣本不能當成該多參考工作負載的效能證據。

## 結論

建議依序研究四項，**全部需要新增程式，不是重新啟用既有快路徑**：

| 排序 | 提案 | 數值分類 | 主要目標 |
|---|---|---|---|
| 1 | FC2 改成完整 1024-channel scale-group TensorOps，搭配專用權重預封裝 | 相對現行 INT8 路徑的精確候選 | 減少 TensorOps 呼叫與資料重取；區分計算、排程與頻寬限制 |
| 2 | FC1＋SwiGLU＋1024-group quantization 真正融合，移除完整 BF16 activated tensor | 相對現行 INT8 路徑的精確候選 | 減少中間張量流量、常駐記憶體與量化 dispatch |
| 3 | QKV 的 norm／RoPE 中間值留在片上，而非同一 kernel 內往返 device memory | 相對現行 INT8 路徑的精確候選 | 消除 Q/K 投影後的中間寫入與重讀 |
| 4 | 僅 FC2 試驗 groupwise W4A8，保持 activation group=1024 | **明確近似、預設關閉** | 降低權重容量與讀取量；不保證提升長序列速度 |

不建議先把全部權重壓成 INT4，也不建議把「去噪很慢」直接解讀成「INT8 GEMM 受 DRAM 頻寬限制」。長序列權重重用高，瓶頸可能是 TensorOps 執行、group epilogue、暫存器壓力或 MPSGraph attention；目前資料尚不能分配各自占比。

## 一、證據範圍與不可變契約

原生來源位置：`ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/`。

委託提供的 snapshot 為 `0f96a9bd859f1576097327e2b7f46298b28d31d8`，binary SHA 為 `e5a88771f646cfc4bcc01b87011255ae10a14c78c47055b92fd37af2ba8689ab`。Astra 直接讀取該目錄檔案，**未另行驗證 commit、binary hash 或兩者對應關係**。

基線沿用選定的 fast4steps：原生解析度、50 層、reuse/core=1、ER-SDE/beta57。四項提案均不得改動 beta5 rank8／time-table、audiovisual joint attention、RoPE、segment／reference 對應、解析度、片長、參考資料、層數與步數。final 必須保持 **BF16 RMSNorm 結果 → F32 modulation → F32 heads**。

直接證據：`encode_forward()` curve-final 分支呼叫 `h3_gpu_adaln_bf16_f32_offset()`，其後以 `h3_gpu_linear_f32()` 執行音訊與影片 heads。`h3_adaln_bf16_f32` 明確先將 normalization 結果 round 到 BF16，再做 F32 scale／shift，並關閉 multiply-add contraction。[S1][S2]

「精確」僅指保持**現行 native INT8 路徑的量化、算術與 rounding 契約**，不表示等同原始 BF16 模型，也不表示已證明 bitwise equality。現行量化是原生 symmetric INT8，不能稱為作者 ConvRot。

## 二、實際程式已做什麼，以及尚留下什麼

### 權重是載入時量化

`quantize_block()` 對每層 FC1、FC2、QKV、attention-out 建立 INT8 tensor 與每 output-channel F32 scales，呼叫 `h3_gpu_quantize_weight_int8()`；同步 submit 後，除診斷保留模式外釋放 BF16 原件。[S3]

權重量化為 `s=max(abs(w))/127`、`q=clamp(rint(w/s),-127,127)`，零值列有 scale fallback。所讀路徑沒有額外 block-packed 權重格式：shader 以 `[output,input]` 邏輯矩陣建立 tensor view，再 slice 給 TensorOps。Morton traversal 是工作群組順序，**不等於權重已預封裝**。[S4][S5]

程序內保留量化權重已經存在；持久化量化權重可能省新程序載入成本，但不是每步長序列成本的主體。新 prepack 必須替換現有表示，不能全模型原始 INT8 與 packed INT8 永久雙份常駐。

### 已有 substantial fusion，不能重複算收益

標準維度 `HIDDEN=5376`、`INNER=7168`、`FFN=14336`：[S6]

| 路徑 | 目前實作 |
|---|---|
| QKV | 長序列可選 full-K=5376、local scales、融合 norm／RoPE、head-major 輸出 |
| attention-out | head-major BF16 直接 gather／quantize 成 row-major INT8，避免獨立 transpose |
| MLP 輸入 | gate＋AdaLN＋INT8 quantization 已融合 |
| FC1 | gate／up 投影與 SwiGLU 已融合；標準形狀可用 full-K=5376 |
| FC2 | activation 每 1024 channels 一個 scale；預設 local kernel K=512、每 group 兩次乘法 |
| 下一層輸入 | MLP gate＋下一層 AdaLN＋QKV quantization 已融合 |
| activation arena | 已有部分 alias，不是全部 tensor 都獨立配置 |

分派見 `h3_gpu_mlp_int8_bf16()`、`h3_gpu_grouped_qkv_linear_rope_int8()` 與 DiT block。[S5][S7][S8]

具體缺口：

1. FC2 的 14 個 activation scale groups 各自形成 INT32 product，再依序累加 F32；預設每 group 仍有兩個 K=512 TensorOps。
2. FC1→FC2 仍寫完整 `[N,14336]` BF16 `activated`，再獨立量化。長序列無 override 時採 scalar grouped quantizer，每 group 先讀一遍求 max，再讀一遍量化。[S7][S9]
3. QKV 雖融合 norm／RoPE，Q/K 投影仍先寫 device destination，再於同 kernel 讀回做 RMS／RoPE並覆寫。[S10]

### 現有計時不能證明 bandwidth 根因

既有四步 log：去噪 wall 413.015 秒、wait 412.794 秒、root-gpu 2.718 秒、200 次 attention dispatch、1640 次 direct dispatch。[S11]

`h3_gpu_stats` 註明 root command-buffer timestamps 不涵蓋所有 MPSGraph child buffers。[S12] 不可把 2.718 秒當完整 GPU 時間，亦不可把其餘 wait 當 CPU 浪費。`h3_gpu_sdpa_graph()` 使用 MPSGraph SDPA；此處不能判定內部是否建立完整 N×N 矩陣。[S13]

**[INFERENCE]** 若時間主要在 joint attention，projection 提案即使成功，端到端收益仍可能有限。

## 三、排序一：FC2 完整 scale-group TensorOps＋權重預封裝

### 設計與 seam

保留 1024-channel activation scales、INT8 權重與 F32 累加順序：

1. 每 group 用一個 K=1024 `multiply`，取代 K=512 `multiply`＋`multiply_accumulate`。
2. 第一版保持 M=128、N=64，避免同時變更多個 tile 維度。
3. 再分別加入 N=128 與 block-packed 權重，以區分收益來源。

這不是重啟 `H3_INT8_GROUP_FC2_LOCAL128`；其仍使用 K=128，且沒有預設 local64 的完整 scale staging 結構。[S5]

- Shader：以 `h3_linear_int8_grouped_local_nax_r128x64()` 為算術基準。
- `h3_gpu.m::h3_gpu_mlp_int8_bf16()`：只替換 grouped-FC2 dispatch。
- `h3_dit.c::quantize_block()`：新增 FC2 packing，不改模型檔或 final。
- `h3_gpu.h`：packed-weight interface 顯式標識格式，禁止當普通 row-major tensor 使用。

候選布局 `[output_tile][k_group][output_lane][k_in_group]`，只排列既有 INT8 bytes，scales 保持 F32且不變，不重新量化。

### 數值、容量與風險

`1024*127^2=16,516,096 < 2^24`；輸入範圍成立時，group INT32 product 及轉 F32 可精確表示。仍須保留 `float(accum)*activation_scale*weight_scale`、group 0→13 F32 加總順序；不得把 weight scale 移到整個 sum 外側或換 pairwise reduction／FMA contraction。

每層 FC2 權重 73.5 MiB；逐層 packing 可把 staging 限一層，不永久增加 50 層約 3.59 GiB 副本。

**[INFERENCE]** K1024 減少 shader 可見 TensorOps 呼叫，不減 INT8 MAC 數；編譯器可能產生相同底層工作，較大 descriptor 可能降 occupancy，prepack 可能破壞 cache locality。

### 待授權最小判別實驗

相同長序列 rows 與輸入：現行 local64/K512 → row-major/K1024 → packed/K1024 → 最後 N64/N128。觀察 kernel及完整 FC2 時間、DRAM traffic、TensorOps utilization、spill／occupancy。只保留實際有效的改動。

成本中高；回退為現行 grouped local64。

## 四、排序二：FC1→SwiGLU→group quantization 真融合

FC1 已消除 `[N,2F]` 投影中間值，不能重複計收益；但 `[N,F]` BF16 activated 仍有 `2NF=28,672N` bytes。**假設** N=32768 即 896 MiB，這不是 log 實際 sequence 長度。[S6][S7]

### 設計

一個 producer 工作群組完成一列的完整 1024-channel group：INT8 gate/up projection → 保留 gate BF16 rounding → 現有 F32 up與SwiGLU → **先 round BF16** → group max → 直接輸出 INT8 activation與F32 scale，不寫完整 BF16 activated。

候選 M16/N1024：BF16 staging 32 KiB。實際 threadgroup memory 加 scratch 若超限，降 M8 或用多個窄 descriptor 共用 group staging。此布局尚未編譯、量測，不能預設有效率。

### alias 是必要正確性條件

現行 `quantized_activation` 同時作 FC1 與 FC2 輸入，靠兩個 dispatch 保序。融合若邊讀邊覆寫，其他工作群組可能尚未讀完，形成資料競爭，**不能原地覆寫**。

安全選項：獨立完整 FC2 INT8 destination，或 row-window FC1→FC2，完成全部消費後才重用 scratch。不得切斷 joint attention或改segment mapping。

若新增 `[N,F]` INT8 destination、刪 `[N,F]` BF16 activated，淨省約 NF bytes，不能宣稱淨省2NF。

### 頻寬與數值風險

移除邏輯 device存取：FC1 BF16寫2NF、max pass讀2NF、emission pass讀2NF，共6NF bytes／層／evaluation。cache可吸收部分，不能等同DRAM收益。

1024 group對齊可能迫使M tile縮小、降低重用；gate/up/activated同時live可能spill；新scratch抵銷容量收益。不可直接量化F32 SwiGLU，否則改變既有BF16→INT8雙重rounding。[S9][S14]

seam為`h3_gpu_mlp_int8_bf16()`與DiT activation allocation。先逐位比group INT8與F32 scales，再比FC2 BF16 output；完整MLP時間也改善才移除正式activated allocation。診斷BF16路徑按需配置。

成本高；回退現行FC1＋獨立quantizer＋FC2。

## 五、排序三：QKV 真正片上 norm／RoPE epilogue

現行`h3_qkv_project_split_int8_rope_local_scales_nax_r128_morton4_impl()`將projection round BF16寫Q/K destination、device barrier、讀回求RMS、再讀norm/RoPE、覆寫。[S10]

新增Q/K專用M64/N128/K5376版本，先寫**16 KiB threadgroup BF16 tile**，保持row dimension 0→127 RMS累加順序，在tile做normalization/RoPE，最後只寫一次head-major Q/K。V維持直接寫出，不付相同staging成本。M64是避免M128的32 KiB tile再加scratch超限，但會增加工作群組，不能預設更快。

精確條件：projection→BF16 rounding、RMS FMA順序與epsilon不變；不新增norm→BF16 rounding（現行norm後F32接RoPE再BF16）；保留48 pairs與其餘維度；head-major index、global row與reference/audio/video mapping不變。

可消除Q/K一次BF16中間寫與兩次邏輯讀，約`12N*INNER` bytes；仍有最終Q/K與V寫入。**[INFERENCE]** cache原已有效或M64降低GEMM效率時可能更慢，須比較projection＋完整epilogue。

seam為`h3_gpu_grouped_qkv_linear_rope_int8()`；封裝Q/K新dispatch與V既有路徑，不散落DiT。成本中高；Q/K/V BF16 buffers逐位驗收；回退現行fused kernel。

## 六、排序四：FC2-only groupwise W4A8，明確近似

僅前述量測證明FC2權重頻寬或容量確有價值後考慮：

- A8維持group1024。
- 從BF16原權重建立每output-channel／1024 K-group symmetric W4，範圍[-7,7]。
- packed nibble載入tile時展開INT8，仍用既有INT8 TensorOps。
- 各group獨立weight scale，原順序F32 accumulation。
- 僅FC2；QKV、attention-out、FC1、AdaLN、final不變。

這是自訂近似實驗，**不是作者ConvRot，也不是已證實M5原生INT4 compute**。現有dtype只有F32/BF16/I8/U32，不可讓I8 tensor暗含nibble語義。[S12]

每層FC2由73.5 MiB降至36.75 MiB payload，50層理論省約1.79 GiB，仍要扣新增scales與bounded unpack scratch；不是全模型減半或footprint保證下降同量。

INT8 MAC數未降、unpack增加指令與片上搬移；每M tile重複unpack可能讓長序列更慢；全模型先展開則喪失容量／頻寬優勢。維持final F32不能消除前50層量化誤差，單層cosine不足判定人物、聲線或嘴型品質。

待授權實驗依序比較W8A8、W4數值預展開I8、同W4數值on-the-fly unpack；前兩者隔離量化誤差，後兩者隔離packing成本。記錄per-group error、尾端誤差、output bias與BF16差異，容忍值必須事前明定。另需授權固定參考圖／音訊四步品質驗收，含identity、音色與AV同步。

成本高；預設關閉；品質或吞吐不合格即回退W8A8，不自動擴張其他projection。

## 七、共用驗收、記憶體與rollout

### 精確候選

以現行INT8為oracle，INT8 bytes、F32 scales、seam BF16 output逐位一致。覆蓋零group、單極值、正負半整數rounding、非128倍數rows、padding、text/reference/audio/video交界與跨request隔離。重排reduction或compiler contraction造成差異，須修正或明列近似，不能默默放寬。「同hidden輸入」下final F32 modulation/heads完全不變。

### 必須區分的成本

- 計算／排程：TensorOps utilization、group呼叫與epilogue。
- 搬移：DRAM、cache、threadgroup、register spill。
- pipeline：MPSGraph attention、完整MLP/block/denoise turnaround。

不能只看kernel數、host encode或root-gpu。

### 約40 GiB原則

packed INT8逐層取代、不留全模型雙份；scratch須有生命週期與byte上限；同時記錄GPU allocations及process footprint。現有26.3 GiB樣本不能把剩餘名義空間全用掉，長片、多reference、framework scratch、常駐LLM都會改變壓力。不停止服務、不假定並行重GPU推論免費。

### 上限僅能作算術

若影響去噪占比p，局部加速s，整體為`1/((1-p)+p/s)`；p未知，不能填projection或attention占比。四步489.602秒、去噪413.015秒，即使全部去噪消失，端到端極端上限約6.39倍，**絕非提案預測**。

rollout：FC2精確改法 → MLP融合 → Q/K片上epilogue → 有條件W4A8。獨立開關與驗收，通過才疊加，不能一次混入fast4steps。

## 原始碼與紀錄索引（Astra實際讀取）

- **[S1]** [`h3_dit.c:2420–2518`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_dit.c:2420-2518)：`encode_forward()` curve-final modulation、F32 heads、output cast。
- **[S2]** [`h3_shaders.metal:3082–3122`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_shaders.metal:3082-3122)：`h3_adaln_bf16_f32()` rounding、scale/shift、contract(off)。
- **[S3]** [`h3_dit.c:728–779`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_dit.c:728-779)：`quantize_block()`。
- **[S4]** [`h3_gpu.m:2873–2880`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_gpu.m:2873-2880)、[`h3_shaders.metal:1332–1401`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_shaders.metal:1332-1401)：weight quantization。
- **[S5]** [`h3_shaders.metal:2688–2896`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_shaders.metal:2688-2896)：grouped FC2 local64/local128、K tiles、scale staging、F32 accumulation、weight view。
- **[S6]** [`h3_dit.c:16–31`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_dit.c:16-31)、[`h3_dit.c:1370–1522`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_dit.c:1370-1522)：dimensions、activation alias、arena。
- **[S7]** [`h3_gpu.m:2980–3185`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_gpu.m:2980-3185)：`h3_gpu_mlp_int8_bf16()`。
- **[S8]** [`h3_dit.c:2004–2149`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_dit.c:2004-2149)、[`h3_gpu.m:3866–3995`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_gpu.m:3866-3995)：joint attention、head-major、cross-block融合、QKV dispatch。
- **[S9]** [`h3_gpu.m:2810–2871`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_gpu.m:2810-2871)、[`h3_shaders.metal:1542–1695`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_shaders.metal:1542-1695)：grouped quantizer分派與重讀。
- **[S10]** [`h3_shaders.metal:1954–2147`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_shaders.metal:1954-2147)：QKV device intermediate、RMS次序、norm/RoPE rounding。
- **[S11]** [fast-four stderr](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/outputs/perf-20261026-memory40-long/fast-four/stderr.log)：既有記錄，未重跑。
- **[S12]** [`h3_gpu.h:10–33`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_gpu.h:10-33)：dtype與root timing限制。
- **[S13]** [`h3_gpu.m:1389–1462`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_gpu.m:1389-1462)：`h3_gpu_sdpa_graph()`。
- **[S14]** [`h3_shaders.metal:2150–2357`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_shaders.metal:2150-2357)：FC1、gate BF16 rounding、F32 up、SwiGLU與BF16 activated。
