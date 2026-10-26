# Astra streaming consultation — 2026-10-26

## Provenance and scope

The following answer was returned by an **actual `functions.astra_consult` call explicitly routed to `llm-esapp/gpt-6-astra`**, supplied the complete `local://h3-quality-safe-contract.md` contract plus the streaming-focused question and installed-source evidence. It is preserved below, not replaced by worker analysis. Read-only research only: no inference, video, build, test, benchmark, runtime/profile changes, service operations, or deployment.

**Subsequent user clarification:** an extremely slight numerical/quality difference may be acceptable, but **no visually/audibly discernible regression in identity, motion, dialogue, voice or lipsync**. Sparse attention, W4A8 and cross-step reuse remain excluded from adoption; any fork/platform proposals using them are research-only. Current-settings authority is [best-settings note](eros-beta5-best-settings-quality-safe-2026-10-26.md); this consultation does not authorize profile changes. Public-fork coverage/commit indexing belongs to the separate Luna survey, not this installed-code streaming analysis.

**Parent-observed current platform/settings evidence (received after consultation):** remote macOS reports `27.0.1 (26A434)` with `128 GiB`; older 26.x performance observations are not guarantees on this runtime. Parent's current-settings note preserves four steps, INT8 projections, BF16 VAE, auto Qwen 8 workers/depth 3, command interval 30 of 50, automatic VAE 320 and automatic zero-copy. Do not force `H3_NAX` or increase VAE tiles blindly. No new settings diff, multiple contexts, duplicated weight sets, enlarged unbounded caches or speculative extra workspace are recommended; retain the approximately 40 GiB soft video budget and LLM/OS headroom. All source changes stay isolated until quality/performance proof.


Worker directly read installed `h3_gpu.m:789–863`, `h3_dit.c:2283–2408,2528–2637,2719–2760`, `h3.c:2190–2275`, prior attention/execution notes, Apple working-set documentation and the FlashAttention abstract. The first attempted SSH hostname `tedmacbookpro` did not resolve; `ssh://ted/` succeeded. Astra used read-only tools for the additional source evidence listed in its answer. Neither source/binary hashes nor claimed runtime behavior were experimentally reverified.

---

## Actual Astra answer (verbatim)

# Astra：H3 工作集串流、可攜性與品質閘門

## 決策

**現在保留既有四步 native INT8 路徑，不開啟 SSD streaming、不更換 attention、不把 ER-SDE 改成 GPU sampler。** 工作集串流有值得研究的方向，但目前沒有證據證明新方案能同時保持既有輸出、降低正式工作峰值並縮短耗時。

優先順序：

1. **不改數值運算的生命週期管理、有限命令窗口與完成後回收。**
2. **保持既有 kernel 歸約及 rounding 的 activation 分塊／融合。**
3. **完整 dense rectangular／online-softmax attention，列為數值候選，不列為 bitexact。**
4. **GPU-state ER-SDE 後置。** 它涉及 solver、RNG、checkpoint 與提交原子性，不是設定開關。
5. **完整去噪後的 VAE／RGB 串流輸出**是另一條可研究的省記憶體路徑；不是去噪中提前產生最終影片。

本次只有唯讀原始碼與官方文件調查；**未執行推論、編譯、測試、benchmark、影片生成、檔案修改或服務操作。** 下列實驗與閘門都是待父代理審閱的提案，不是已通過驗證。

先前研究中的 sparse/local attention、W4A8、跨步 KV/residual reuse，**在本次品質契約下全部排除**，不保留為隱含後備方案。

---

## 一、目前實際建議設定

video-maker [`profiles/h3.json`](https://github.com/mics8128/video-maker/blob/main/profiles/h3.json) 已明確選擇以下設定。這是已讀到的設定檔，不代表本次另行確認了遠端程序環境或實際 dispatch。

| 項目 | 現在建議 |
|---|---|
| Sampler／scheduler | `er_sde`／`beta57` |
| 步數 | `4`；保留有效 schedule、terminal zero 與原先去重規則 |
| Layers | `50` |
| Denoise／core reuse | `1`／`1` |
| 解析度 | 原生 `768×1344`；橫向依既有流程交換，不增加低解析度中間生成 |
| 片長／素材 | 保持各工作的實際片長、全部 references／guides；已準備的 141 幀工作不縮短 |
| Projection | 既有 native M5 INT8；`bf16_projections:false` |
| Video VAE | `bf16`，保留既有 spatial tile、temporal chunk 與 blending |
| Audio VAE／sampler state | 保留既有 F32 路徑 |
| Video／audio shift | `12/3`；這是目前 native fallback，不宣稱作者 beta5 專屬最佳值 |
| Beta5 final | **BF16 RMSNorm → F32 modulation → F32 heads**，之後保留既有輸出 BF16 cast／F32 widening |
| Attention | 現行完整 noncausal joint MPSGraph SDPA |
| SSD streaming | **關閉** |
| 命令切分 | 保留既有預設；不新設激進 `H3_DIT_COMMAND_BLOCKS` |
| GPU sampler window | 不拿 `H3_GPU_SAMPLER_WINDOW` 當 ER-SDE 優化設定 |
| 記憶體 | 約 `40 GiB` 軟目標，保留 LLM／OS 空間；不把剩餘額度全部配置成 scratch |
| 執行方式 | 保持目前工作入口；不為此次提案切換常駐服務或增加重 GPU 併發 |

設定檔中的 `seconds:4` 是預設值，**不是把已準備的 141 幀／5.875 秒工作改回四秒的指示**。

### 三個重要實作細節

**1. SSD streaming 會改變目前 projection 路徑。**

已安裝來源把 `int8_mlp`、`int8_qkv`、`int8_attention_out` 全部綁定在 `!dit->ssd_streaming`。因此現在開 SSD streaming，不只是把相同 INT8 權重換位置，而會停用這三條 INT8 路徑。這不符合「優先保持現行 native INT8 輸出」的要求。[S2]

**2. 現行四步 ER-SDE 已有命令切分，不是每層 CPU 等待。**

50 層時，預設 `command_block_interval()` 為 30；第 30 層後 `continue()` 提交，forward 尾端 `submit()` 等待。依這條程式路徑，每次 evaluation 通常是兩個應用層命令提交；不是把 200 個 block 都累積到最後才等，也不是每個 block 都等。[S2][S4]

**3. 「F32 final」不代表 sampler 直接收到未經 BF16 rounding 的 head 輸出。**

Beta5 final modulation／heads 的確是 F32；但現行程式隨後將輸出 cast 為 BF16，CPU 再逐位 widen 成 F32，才 unpatchify／unpack。GPU-state 改造若直接使用 F32 head 結果，反而會改掉目前 oracle。[S3][S8]

---

## 二、先把「streaming」分成四件事

| 種類 | 串流的是什麼 | 已有狀態 | 能否因此提前得到最終影格 |
|---|---|---|---|
| SSD weight streaming | SSD → 兩個 layer weight slots | 已有；逐層 prefetch、GPU 完成後輪替 | 不能 |
| Activation／attention working-set streaming | query、MLP 中間值、attention tiles 等短生命期資料 | 已有 arena 重用／alias；更細分塊尚待實作 | 不能直接推得 |
| GPU command streaming | CPU 持續編碼、提交有限 GPU 工作並退休已完成資源 | 已有 `continue()`；沒有一般化有界 completion retirement | 不能 |
| Final-frame output streaming | 去噪後，逐段 VAE decode、RGB 轉換／交付 | VAE 已分塊運算，但目前仍累積完整 RGB 後交付 | **能研究去噪後較早交付，不是去噪中最終成品** |

這四件事應分開驗證。SSD 讀取、Metal command submission、attention tiling 都不能自動稱為「邊生成邊輸出最終影片」。

---

## 三、已經存在的工作集優化

### 3.1 Activation arena 已跨層重用

`allocate_activations()` 已預配置 hidden、Q/K/V、attention、MLP、量化 scratch 與 final scratch。預設 alias 包括：

- `attention_heads` 與 `mod_mlp` 共用 `qkv` arena。
- MLP output 在執行路徑中重用 attention output。
- 跨 block 已有 gate／AdaLN／INT8 quantization 融合。

所以不能把「每層不再配置一份 activation」當作全新收益，也不能看到 fused QKV 不再需要原始 QKV 輸出，就直接刪掉 `qkv` buffer；它仍負責後續 scratch。[S1]

### 3.2 SSD 路徑已經雙槽預取

現行 SSD 分支：

1. 使用已就緒 slot 的 layer weights。
2. `pthread` 預取下一個 active layer 到另一 slot。
3. 執行目前 block。
4. GPU `submit()`／等待完成。
5. join 預取、檢查錯誤、交換 slot。

SSD 模式另外關閉一般 command splitting。這是已有的權重容量方案，不是待補的 activation streaming，也不是目前 resident INT8 路徑的效能歸因。[S2]

### 3.3 Framework attention workspace 仍不透明

現行 DiT 使用 BF16、noncausal MPSGraph SDPA；原生 wrapper 的 Q/K/V 共用同一個 sequence shape。官方 SDPA 契約定義：

\[
O=\operatorname{softmax}(sQK^T+M)V
\]

但未承諾內部 tile、softmax 累加精度、workspace 大小或所有 shape 的 kernel 選擇。[S5][P1]

**沒有證據證明目前會常駐完整 \(N\times N\) score matrix。** 新 tiled attention 不能拿「消除原本完整平方矩陣」當作已證實收益。

---

## 四、Query tiling 與 K/V tiling：完整 dense，不等於逐位一致

### 4.1 Query-only tiling：較小的第一個數值實驗

對 query 分塊：

\[
O_I=\operatorname{softmax}(sQ_IK_{\mathrm{all}}^T)V_{\mathrm{all}}
\]

每個 query 仍看到全部文字、guides、references、target audio、target video。不同 query rows 的 softmax 彼此獨立，因此這種切法**不減少模型 attention 邊**。

Apple SDPA 文件本來就允許 `Nq` 與 `Nkv` 不同；目前 H3 wrapper 則只有單一 `sequence`，需要內部 rectangular adapter，不能只改呼叫參數。[P1][S5]

具體限制：

- 保持原 packed row 次序、segment spans、RoPE、scale 與 mask。
- `[H,N,D]` 的 query 子區塊跨 heads 帶有 `N×D` stride；**不能只加一個 buffer offset，就當成連續 `[H,Bq,D]`**。
- 使用可證明正確的 slice／stride，或有界 gather/scatter；成本全部計入。
- 每個 query tile 都使用同一份完整 K/V，不逐塊複製整份 K/V。
- Graph cache key 必須區分 query／KV 長度、layout、dtype、scale 與尾塊形狀。

**品質分類：數值候選。** 即使數學上每 row 的公式完全一樣，MPSGraph 仍可能因矩形 shape 換 kernel、歸約樹或 precision path。未測之前不能標為 bitexact。

**[INFERENCE]** 可先以 `Bq=512/1024` 作小範圍判別；這是實驗起點，不是現時推薦設定。它可能降低 workspace，也可能因 dispatch、重讀或 gather/scatter 變慢。

### 4.2 K/V tiling：必須保持單一 full-joint softmax

對固定 query tile，完整掃描所有 K/V tiles。每 row 維持 F32：

- \(m\)：目前最大 score。
- \(\ell\)：正規化分母。
- \(u\)：未正規化 weighted value accumulator。

每個 tile 的更新：

\[
\begin{aligned}
S &= sQK_t^T\\
m' &= \max(m,\operatorname{rowmax}(S))\\
a &= \exp(m-m')\\
P &= \exp(S-m')\\
\ell' &= a\ell+\operatorname{rowsum}(P)\\
u' &= au+PV_t
\end{aligned}
\]

最後才計算 \(O=u/\ell\)，並按既有契約寫回 BF16。

這是 FlashAttention 類完整 dense IO tiling 的核心：避免把完整 score／probability 寫到外部記憶體，**沒有消除 \(O(N^2)\) attention 算術**。論文中的 exact 是非稀疏、非截斷的 attention，不是與特定 MPSGraph 實作逐位相同。[P2]

禁止：

- 各 K/V tile 各自 softmax 後平均輸出。
- Audio、video、reference 分別 softmax 再相加。
- 丟棄「小概率」keys。
- 為 TensorOps 方便，未經單獨審查便將 F32 \(P\) 降為 BF16。
- 在不改名稱的情況下加入 local／causal mask。

尾塊必須遮蔽無效 keys、避免 padding V 被讀取，並處理空有效區間的 `-∞−(-∞)`。每個合法 key 必須恰好參與一次。

### 4.3 Split-KV 不列為首選

如果後續把 K/V 分成多個並行 partition，必須以各 partition 的 \(m_s,\ell_s,u_s\) 做穩定合併，而非合併已獨立正規化的 output。

它增加 partial accumulators、額外歸約與記憶體流量；完整長序列本來已有大量 query/head 工作，不能預設缺乏並行度。

**決策：先做單一完整 K/V 掃描候選；只有量測指出長迴圈或 occupancy 問題，才研究 split-KV。** 新增 scratch 必須納入 40 GiB 目標。

---

## 五、Activation／layer working-set streaming 的安全邊界

### 可優先研究的方向

**[INFERENCE]** Row-local MLP、output projection、部分 norm／quantization，可以用較小 row windows 執行，再依原順序寫回。這有機會降低全序列中間張量或其讀寫，但必須同時保持：

- INT8 group 定義、scale 計算、量化 rounding。
- FC2 的 K-group 順序與 F32 累加順序。
- 原 BF16 materialization 邊界，即使不再真的寫入全量 buffer。
- RMSNorm 的原歸約樹。
- SwiGLU 的相同運算與 rounding。
- Head-major layout 與既有 padding 行為。
- 跨 block fused AdaLN／量化的 producer-consumer 次序。

對現行 INT8 oracle 而言，「更高精度」或「數學等價」都不是略過 bitwise gate 的理由。

### 不能把完整 transformer 按時間片獨立跑完

同一層 query chunk 仍依賴該層全部 K/V；下一層 K/V 又依賴前一層全部相關 hidden 結果。

因此安全的基本排程是：

1. 確立本層完整輸入／所需 K/V。
2. 分塊完成本層 query 與 row-local 運算。
3. 保留尚有讀者的資料。
4. 下一層開始前滿足其全域依賴。

不能把「時間片 A」先跑完 50 層，再處理 B，卻宣稱仍是原本 noncausal joint attention。

References 也不能例外：固定的是原始素材，不是每層 reference hidden。Joint attention 會讓 reference hidden 接收 target 資訊；跨層或跨步重用它的 K/V，會改模型。

### Unified memory 不等於免費 offload

目前 CPU/GPU 共享記憶體環境下，移到另一個 host buffer 通常只是多一份資料與搬運；它不自動減少整個程序 physical footprint。Purgeable empty 更不是「日後可以原值讀回」的 offload。若要丟棄，就必須確定不再需要，或有經審查的原值重建途徑。[P5]

**決策：先縮短暫存生命期，不先導入 activation SSD spill 或多份 host/device backing。**

---

## 六、Bounded command windows 與完成後回收

### 現況

`h3_gpu_begin()` 拒絕存在 active command 或 inflight commands 的狀態。

`h3_gpu_continue()`：

- commit 目前 command；
- 加入 `inflightCommands`；
- 開始下一個 command；
- 不退休已完成項目。

`h3_gpu_submit()`：

- commit 尾端 command；
- 等所有 inflight；
- 檢查錯誤；
- 清空陣列。[S4]

所以它是**可串接、尾端整批 drain**，不是一般化的有限完成窗口。但在目前預設 ER-SDE 中，每個 forward 都 drain，不能據此宣稱已累積無界的四步命令數。

### 建議的內部設計

保留既有公開 `begin/continue/submit` 語義，先不增加 exported symbols：

- 每次提交建立內部 completion ticket：command、需保活資源、最後使用關係、完成狀態與錯誤。
- 有界窗口滿時，只等待／退休最舊可退休工作，而非一律 drain 全部。
- `submit()` 仍是完整 drain 與錯誤回報點。
- CPU 讀結果、checkpoint、slot 覆寫與資源回收，依其實際最後使用的 completion 判定。
- GPU 錯誤後不在未知狀態自動回退並重新生成。

**[INFERENCE]** 第一輪可比較一個與兩個應用層窗口，但不直接改成預設；目前預設本來約兩個提交／evaluation，收益未必大。

Apple 要求 completion handler 在 commit 前註冊，handler 在 GPU 完成該 command buffer 後執行；也適合檢查 `status/error`。它不是「提交即完成」通知。[P3][P4]

### 最危險的生命週期問題

`h3_gpu_tensor_free()` 會立即：

```text
setPurgeableState(MTLPurgeableStateEmpty)
```

Apple 定義 `empty` 為內容無效／丟棄。[S4][P5]

因此：

> **Command buffer 保留 buffer 物件，不等於可以提前將 buffer 內容設為 empty。**

退休條件必須涵蓋所有最後讀者／寫者及 aliases；不能在 encode 完成、commit 完成、scheduled callback，或某個較早 command 完成時就 purge。

若 shared activation arena 在較新 command 中仍有讀者，較舊 command 的 completion 只能退休其專屬資源，不能釋放整個 arena。

### MPSGraph child work 的限制

H3 將 MPSGraph 編碼到 `MPSCommandBuffer`，並取回 `rootCommandBuffer`。官方文件沒有在已讀頁面中給出足以推定所有內部 child workspace 壽命與 peak 的細節。[S5][P8]

因此：

- 窗口上限先只稱為**應用層命令上限**。
- 不宣稱同時限制全部 framework child commands／workspace。
- 不拿 root GPU timestamp 當整段 GPU 工作時間。
- 回收必須維持 graph dependency 所需的完整完成邊界。
- Completion handler 與 encoder 執行緒對 retirement queue／error state 的存取需要明確同步，避免 use-after-free 或死鎖。

---

## 七、GPU-state ER-SDE：可行的研究方向，不是現在的最佳設定

目前 `H3_GPU_SAMPLER_WINDOW` 只服務 `denoise_euler_gpu()`；不會替已選 ER-SDE 啟用 GPU-state。[S3]

### 必須原樣保存的狀態與時序

| 契約 | 移植要求 |
|---|---|
| Packed AV clocks | 令 \(c=video\_shift/audio\_shift\)，保留 \(r=c+(1-c)\sigma_v\)；audio carried state 是 `r × native_audio` |
| Clean prediction | Video：`native + sigma_v × velocity`；audio：`c × (native + sigma_a × velocity)` |
| Solver | 保留 lambda／phi、200 點 left-endpoint quadrature、各 transition coefficient 計算次序 |
| Histories | `old` clean prediction 與 `old_derivative` 都需保留；不只保留上一個 velocity |
| Start age | 使用 `step-start_step`，不假設永遠從 transition 0 開始 |
| Terminal | `next==0` 複製 prediction；不多抽 noise |
| Masks | 保持 hard／soft mask 投影次數與位置；不對 prediction 和 proposal 重複套 soft mask |
| 數值邊界 | 保留 BF16 model velocity rounding 後再 widen 至 F32 |
| Commit | Proposal／finite checks 成功後才提交 histories、RNG、native/carried 與 step |
| Observe／pause | Observer 看見的是已提交 transition；取消不暴露半更新狀態 |

以上可直接對照目前 CPU oracle。[S6]

### RNG 不能只看 seed

原生初始 video 與 audio noise 使用**各自同 seed 的 generator**；ER-SDE transition noise 則按 video 再 audio 的順序推進 sampler RNG。

目前 transition Gaussian 使用 SplitMix64 與 CPU double `log/sqrt/sin/cos`。改成 GPU float transcendental、平行重排抽樣，或改變奇數長度時的 pair 消耗，都可能改變整條軌跡。[S6][S7]

保守候選應：

1. 保留現有 CPU coefficient 與 RNG 產生方式。
2. 將同一份 F32 coefficients／noise 交給 GPU。
3. 先只搬 state elementwise update。
4. 保留未提交 proposal，驗證成功後再交換 committed state。
5. 只有這條路徑逐位通過，才討論再搬 RNG。

它仍可能受 CPU／GPU FMA、除法、運算收縮差異影響，不能先宣稱 bitexact。

### 初始化與 checkpoint

`h3.c` 在 ER-SDE first-sigma offset 前保存 initial noising 用的 sigma；模型 schedule 才使用 offset 後的版本。不可用新 effective sigma 回頭重算 initial noising。[S7]

ER checkpoint 仍須保持：

- solver `3`／revision `1`。
- 七份 arrays：native、carried、clean history、anchors、masks、preservation noise、derivative history。
- 原 schedule、RNG、step/history 與 request/backend fingerprint。
- Snapshot 只能取已完成且已提交狀態。[S6]

**格式相同不代表新 backend 可以直接讀舊工作繼續 exact resume。** 不應繞過既有 dependency fingerprint；跨版本相容需要另外證明。

**判斷：**現有證據不足以把 CPU sampler 搬移排在 attention／MLP 前面。這是高正確性風險、收益尚不明的後置工作。

---

## 八、最終影格能不能在完整去噪之前輸出？

### 現有路徑：不能

實際順序是：

1. 完成 joint AV denoise。
2. 釋放非 cached DiT。
3. Decode 完整 audio。
4. Decode video latent。
5. 完整 F32 RGB 轉 U8。
6. 才逐幀呼叫 output callback。[S9]

現有 denoise preview 是中間結果，不是最終影格。

### 為什麼不能按時間切片先完成前半段？

Noncausal joint attention 讓早期影格 latent 依賴後段 video、audio 及其他 packed tokens；後續層與下一個 denoise transition 都可能改變它。

把前半段 latent 冻結、局部去噪或提前 decode 後不再更新，會改掉原模型。這不在此次品質契約內。

**[INFERENCE：狹窄例外]** 在最後一次 evaluation 的最後一層，若完整全域 K/V 已固定，某些 query rows、row-local final head、terminal solver proposal 可以先完成。但要據此提前發布最終影格，仍須處理該 VAE chunk 所需全部 latent、全工作失敗／取消、mask 與提交驗證。現有程式沒有這條管線；還會讓 DiT／VAE 同時駐留並競爭 GPU。**不建議現在追求。**

### 真正可研究的是「去噪後、整部 VAE 尚未完成」的最終影格串流

已安裝 video VAE 並非一次對完整時間序列做無界 decode：

- 每 chunk 取 **7 個 latent time slices**。
- 下一 chunk 前進 **5 個 latent slices**。
- 每 chunk decode **22 frames**。
- 相鄰 chunk 有 **5-frame temporal overlap**。
- 每輪確定 **17 frames**，最後補尾端 5 frames。
- Spatial tiles 另有固定拼接與 blending。[S10]

因此 **[INFERENCE]** 可以研究：

1. 完整 denoise 後開始既有 VAE。
2. 完成一個 chunk 的全部 spatial stitching。
3. 以原次序完成與前 chunk 的 temporal blending。
4. 將已確定的 17 frames 轉 U8 並交給有界 sink。
5. 保留尾端 5 frames；最後一輪再交付它們。

目前程式仍配置完整 `final_rgb`。141 幀、768×1344 的完整 F32 RGB payload，依尺寸計算約 **1.63 GiB**；這是配置尺度計算，**不是已量測可節省的峰值**。Sink 若又累積整部影片，收益會消失。

品質要求：

- 不改 7/5 latent window、22/17/5 frame 規則。
- 不改 spatial tile、frame mapping、overlap alpha 或 blending 次序。
- RGB F32／U8 必須逐位對照既有 decoder。
- 保持 audio sample count、24fps frame timestamps、mux 與取消語義。
- 已交付部分 frames 不代表工作可以提前標為成功。
- Raw frames 相同也不自動證明 MP4 bytes 相同；encoder／mux 行為須分開驗證。

這條路徑較適合處理輸出記憶體與首個最終影格的交付時間，不能當作 denoise 加速。

---

## 九、可攜性閘門

### 9.1 分開判斷「能執行」與「符合目前品質基準」

目前 H3 以裝置名稱包含 M5、TensorOps library 編譯成功等條件選路；TensorOps 編譯失敗可回到一般實作。[S11]

這種 fallback 可以提供功能可用性，但**不代表與 M5 native INT8 oracle 相同**。若其他裝置最後跑 BF16 projection，應清楚標示 backend 改變，不可把它列為本次 exact qualification。

### 9.2 新 kernel 需檢查的能力

- OS／SDK／Metal language 與所需 TensorOps 功能。
- 實際 GPU family。
- BF16 與 matrix operation 支援。
- `maxTotalThreadsPerThreadgroup`、thread execution width。
- Static＋dynamic threadgroup memory。
- Buffer 長度、offset／alignment、尾塊與 address arithmetic。
- 相同 layout／dtype／scale 是否有已驗證路徑。

Apple feature table 將 M5 列為 Apple10；列出的 compute threadgroup memory 上限為 32 KB。**不能因為 M5 更新，就假設可以把 Q/K/V、scores、F32 output accumulator 全放進 threadgroup memory。** 例如 `32×64` attention tile 的單份 BF16 K 與 V 就各約 16 KiB，合起來已達 32 KiB；還沒計其他資料。[P6]

應按實際 pipeline 限制配置 register／staging。較大 tile 可能 spill；不先宣稱更快。

### 9.3 保留現行數學模式

目前 library 使用 `MTLMathModeSafe`；beta5 final shader 還明確保留 BF16 roundtrip，並關閉該段 multiply/add contraction。[S8][S11]

新候選不能順便開 fast math、改 approximate exp、改 RMSNorm tree，然後將差異都歸為「streaming」。

### 9.4 能力回退與錯誤回退不同

- **Dispatch 前**發現不支援：可選已存在、事先驗證的 baseline。
- **GPU 執行後**失敗：保留診斷並回報，不在未知結果下重新送整個工作。

可攜性驗證需記錄 device、OS、compiler、backend 身分；單一裝置通過，不宣稱跨裝置逐位等價。

---

## 十、品質與整合閘門

### Gate 0：固定 oracle 與工作契約

在隔離 clone 開發；已安裝 runtime 保持 untouched oracle，模型唯讀。

固定：

- 4-step ER-SDE/beta57、50 layers、reuse/core1。
- Native 768×1344、原片長及全部素材。
- INT8／BF16／F32 邊界、RoPE、packed segments、RNG 與 init order。
- 原始 compiler/math mode、執行 backend 身分。
- 不同候選分開改，不能 attention、quantization、sampler 一次混合比較。

141 幀 target 本身有：

\[
42{,}336\ \text{video tokens}+470\ \text{audio rows}=42{,}806
\]

正式 \(N\) 還須加文字、全部 references／guides。來源 temporal/layout 規則見 [S12]；**約 45k rows 只是尺度示例，不是已測到的多角色工作 \(N\)**。

### Gate 1：分類，不准把誤差重新命名成無損

| 類型 | 通過條件 |
|---|---|
| 純排程／lifetime 改造 | 同輸入、同 backend 的 committed tensor／state bytes **零差異** |
| 宣稱 exact 的 INT8 fusion／prepack | Quantized bytes、scales、BF16／F32 公開邊界 **零差異** |
| Rectangular／tiled attention | 一旦非零差異，就列為數值候選；不得標 bitexact |
| GPU-state ER-SDE | Native/carried/history/RNG／transition 與 checkpoint 語義逐項對照 |
| VAE output streaming | 原 chunk／stitch 結果與 RGB U8 逐位一致；影音時間軸一致 |

所有路徑都要求：

- 無新增 NaN／Inf。
- 無越界、padding 污染、alias 覆寫、錯誤素材／speaker 對應。
- 錯誤或取消後沒有半提交 checkpoint。
- 符合既有 request ownership，不能 A→B→A 串料。

### Gate 2：完整 dense attention 的數值審查

父代理若批准非影片測試，應以同一份 BF16 Q/K/V 比較候選與現有 MPSGraph，另用小尺寸高精度 oracle 檢查公式。

至少涵蓋：

- Production `56 heads、D=128`。
- 非 tile 整數倍、邊界及長序列。
- 平坦 logits、極集中 logits、大動態範圍與 cancellation。
- Head-major gather/scatter。
- 原始 global K/V 覆蓋與 tail mask。
- 全部 rows／heads，而非少量抽樣。

報告：

- Absolute error、帶近零處理的 relative error。
- BF16 ULP 分布及最大值。
- 每 row／head 最壞值、RMS error。
- Downstream projection、完整 block 的誤差傳播。

**沒有一個普遍適用的 `atol` 或 PSNR，可以證明「極輕微成品影響」。** 容忍值必須在看候選結果前登記；失敗後不放寬。若只通過 kernel 數值測試，仍不能開啟正式預設。

### Gate 3：非影片的狀態／生命週期證據

由父代理決定並集中執行，這次沒有跑：

- 長 command chain、窗口滿／未滿、完成次序與最後讀者。
- CPU read／slot reuse／purge 必須在完成後。
- Graph＋direct Metal 混合依賴。
- Error／cancel／partial failure 的 drain 與診斷保留。
- ER-SDE start-step、每個 pause boundary、terminal no-noise。
- 奇數 noise 長度、RNG progression、hard／soft masks。
- Solver3/revision1 save/load 與未中斷狀態對照。

### Gate 4：效能與記憶體

只有 Gate 1–3 過關，才值得批准 bounded microbenchmark：

- 使用實際 \(N\) 與 layout，而非只測 2048 rows。
- 分開首次 compile／steady state。
- 計入 gather/scatter、prepack、extra dispatch、workspace。
- 分別比較 kernel、完整 block、排程窗口；不把局部收益直接乘 200。
- 分開列 physical footprint、Metal resource allocation、framework workspace 與系統 pressure／swap；不同口徑不可直接相加。

Apple 的 `recommendedMaxWorkingSetSize` 是裝置資源不影響效能的大致建議，**不是影片程序配額，也不是替 LLM 保留 RAM 的機制**。[P7]

40 GiB 是軟目標。小幅超出需揭露，但不以降解析度、少 references 或少 frames 自動「修復」。保留 LLM 權重空間，也不代表可免費同時執行重 GPU LLM 推論。

### Gate 5：使用者解除影片暫停後，才有完整品質審核

數值候選要成為預設，仍需原設定的完整四步影音驗收：

- 雙角色／全部 references。
- 人物身分與說話者綁定。
- 台詞、音色、嘴型與 AV 同步。
- 遮擋、轉身、後段穩定性。
- Temporal VAE 接縫、首尾 frames。
- 多個事前選定案例／seed，而非只挑成功樣本。

**目前影片暫停，因此這道 gate 尚不能通過。** 完成非影片測試可支持保留研究分支，不能替代正式品質批准。

---

## 十一、效能證據的界線與建議整合順序

既有單圖＋單音訊、141 幀／5.875 秒樣本：

| 指標 | 已提供的既有觀察 |
|---|---:|
| 總時間 | 489.602 s |
| Denoise | 413.015 s |
| Video VAE | 53.973 s |
| 0.25 s 間隔 sampled physical footprint | 26.309 GiB |

它是非獨占、單樣本，**不是雙角色五圖兩音訊的正式量測**；抽樣峰值也不保證捕捉短暫峰值。離線 prepare 通過，只證明準備路徑，不證明執行時間、完整 \(N\)、峰值或品質。

Denoise wait 包含 GPU／MPSGraph 工作；root GPU timestamps 不涵蓋所有 child work。不能把兩者相減後稱為可刪除的 CPU 空轉。

### 審閱後的建議次序

1. **保持現在設定與部署不動。**
2. 在 clone 提出 bounded retirement／alias lifetime 的局部改動；保持公開介面與運算不變。
3. 若要縮 activation，先選可保持原運算順序的 row-local 路徑；與 attention 改造分開。
4. Rectangular SDPA 作小型數值／workspace 判別；native online-softmax 作另一獨立候選。
5. 去噪後 VAE/RGB sink 可作獨立省記憶體提案，不綁定 denoise 改造。
6. GPU-state ER-SDE 後置，除非後續證據顯示其資料搬運／排程值得承擔相容性成本。
7. 父代理集中執行獲准的最終檢查；整合審閱後再決定是否採用。**本提案不授權部署、發布或恢復影片生成。**

**最終建議：工作集串流可以在不降低模型品質的方向上研究，但「現在安全可用」的是既有 arena／融合／命令切分，不是尚未驗證的新 streaming 開關。完整 dense tiling 是合理數值候選；真正較低風險的起點是生命週期與保留運算次序的暫存縮減。最終影格串流則應先限定於完整去噪之後。**

---

## 主要來源

### 已安裝原始碼

以下為本次讀取的已安裝來源；行號用於定位，不宣稱本次重新驗證 source／binary hash 一致。

- **[S1]** [`h3_dit.c:1369–1567`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_dit.c:1369-1567)：activation arena、aliases、INT8 scratch。
- **[S2]** [`h3_dit.c:213–224`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_dit.c:213-224)、[`1748–1788`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_dit.c:1748-1788)、[`2283–2408`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_dit.c:2283-2408)：命令預設、SSD 排除 INT8、雙槽預取。
- **[S3]** [`h3_dit.c:2488–2637`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_dit.c:2488-2637)、[`2708–2768`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_dit.c:2708-2768)：F32 heads 後 cast、CPU pack/readback、GPU Euler window。
- **[S4]** [`h3_gpu.m:706–719`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_gpu.m:706-719)、[`789–863`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_gpu.m:789-863)：purgeable free、begin/continue/submit。
- **[S5]** [`h3_gpu.m:1391–1554`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_gpu.m:1391-1554)：SDPA shape/cache、noncausal graph、MPS command encoding。
- **[S6]** [`h3_sampler.c:12–175`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_sampler.c:12-175)、[`176–469`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_sampler.c:176-469)：RNG、packed AV、ER-SDE、commit、checkpoint。
- **[S7]** [`h3.c:1259–1277`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3.c:1259-1277)、[`2132–2159`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3.c:2132-2159)：preoffset initial sigma、分開初始化 AV RNG。
- **[S8]** [`h3_shaders.metal:3081–3125`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_shaders.metal:3081-3125)、[`h3_dit.c:2425–2439`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_dit.c:2425-2439)：beta5 BF16 RMSNorm／F32 modulation。
- **[S9]** [`h3.c:2194–2293`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3.c:2194-2293)：denoise、audio/video decode、RGB 與 frame callback 次序。
- **[S10]** [`h3_video_vae.c:873–1000`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_video_vae.c:873-1000)、[`1090–1156`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_video_vae.c:1090-1156)、[`1164–1290`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_video_vae.c:1164-1290)：spatial stitch、7/5 latent chunk、22/17/5 frame assembly。
- **[S11]** [`h3_gpu.m:340–417`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_gpu.m:340-417)、[`440–462`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_gpu.m:440-462)：Safe math、M5/TensorOps 選路與 fallback。
- **[S12]** [`h3_host.c:18–38`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_host.c:18-38)、[`248–295`](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_host.c:248-295)：temporal mapping、packed layout row accounting。

### 平台與演算法第一手文件

- **[P1]** Apple：[MPSGraph scaled dot-product attention](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph/scaleddotproductattention(query:key:value:mask:scale:name:))。
- **[P2]** Dao et al.：[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)。
- **[P3]** Apple：[MTLCommandBuffer](https://developer.apple.com/documentation/metal/mtlcommandbuffer)。
- **[P4]** Apple：[addCompletedHandler(_:)](https://developer.apple.com/documentation/metal/mtlcommandbuffer/addcompletedhandler(_:))。
- **[P5]** Apple：[MTLPurgeableState.empty](https://developer.apple.com/documentation/metal/mtlpurgeablestate/empty)。
- **[P6]** Apple：[Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)，GPU families、功能與 threadgroup limits。
- **[P7]** Apple：[recommendedMaxWorkingSetSize](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize)、[currentAllocatedSize](https://developer.apple.com/documentation/metal/mtldevice/currentallocatedsize)。
- **[P8]** Apple：[MPSCommandBuffer](https://developer.apple.com/documentation/metalperformanceshaders/mpscommandbuffer)、[rootCommandBuffer](https://developer.apple.com/documentation/metalperformanceshaders/mpscommandbuffer/rootcommandbuffer)。
