# Eros beta5：Luna → Astra 固定來源移植與品質安全評估

**評估者：`llm-esapp/gpt-6-astra`。本報告承接指定 Luna 調查，並以固定版本原始碼及已安裝 runtime 比對，不是另一輪 fork 排名。**

## 一、結論

1. **最值得研究的去噪串流路徑，是每個 query tile 讀取完整 K/V 的 dense attention。**近期從 PR77 式 MPSGraph query slicing 開始；需要真正控制工作區生命週期時，再改為獨立 rectangular dispatch。它們都不能直接宣稱更快或更省記憶體。
2. **macOS 27 TensorOps cooperative-tensor online softmax 是可行性研究，不是待開啟的現成功能。**主程序提供的 OS、SDK 與動態 Metal 編譯證據，足以排除「主機仍停留在 macOS 26」的錯誤前提；仍未證明特定 cooperative tensor／matmul 組合能編譯、保持所需精度及有效執行。
3. **目前最小、最容易維持原算術的增量，是修正 `h3_gpu_mlp_int8_bf16()` 對十個 pipeline 的提前索取。**中央 PR75 lazy getter 已存在；不能再算成一次 PR75 移植，也不能沿用上游 14% 宣稱。
4. **jerryfane Conv3d arena 確有尚未被本機涵蓋的 allocation／submission 差異；效益在參考圖編碼，不在 DiT 去噪。**只取保留 Conv3d、padding、norm、權重與 tile stitching 的部分，不取 Conv2d 替換。
5. **DiT-load cache 與 PR55 必須分開看。**前者主要減少跨程序重複載入／量化；後者以逐層權重 I/O 換常駐記憶體，**不是 KV cache**。兩者原版均不足以直接接上目前的動態 LoRA 與 beta5 rank8／F32 final 契約。
6. **沒有任何新來源取得本機加速認證。**影片暫停期間，即使 attention 數值門檻通過，也不能認證身份、動作、中文音色／台詞、嘴型及影音同步無可察覺退化。

### A0–A6 決策總表

「可採用」在此指可進入隔離實作與驗收，不代表已部署或本報告批准變更正式設定。

| ID | 決策 | 相對已安裝 baseline 的實際差異 |
|---|---|---|
| **A0：既有 dense MPSGraph 歸因** | **既有；量測解讀須補強** | `h3_gpu_sdpa_graph()`、`h3_gpu_sdpa()` 已存在。`h3_gpu_submit()` 累計所追蹤 command buffer 的 GPU 時間，不能由此斷言包含全部 MPS child work，亦不能識別內部 score workspace。 |
| **A1：PR64／PR77 rectangular query** | **研究門檻；優先去噪候選** | 本機仍為完整 Q/K/V graph。固定 PR 以 graph slice 切 Q，K/V 不切，最後 concat；不是已經具備受控 scratch 的逐塊 host streaming。採 PR77 的 opt-in／安全尾塊邏輯，不採 PR64 預設啟用。 |
| **A2：TensorOps online softmax** | **研究門檻；暫不預設採用** | Apple 提供 cooperative tensor reduction 與直接餵入下一個 matmul 的 API；本機未證明所需 shader 組合。原生 INT8 projection TensorOps 已很多，新增價值在 attention，不是泛稱「用 tensor cores」。 |
| **A3：Conv3d arena** | **可採用其等算術設計，須 byte gate** | 本機 `run_conv()`／`run_block()` 反覆配置 scratch 並同步提交；來源 `prepare_planar_arena()`／`planar_conv()`／`encode_tile_planar()` 重用 arena、分段 continue、每 tile 等待。只適用其單幀路徑。 |
| **A4：DiT-load cache** | **研究門檻；拒絕原版直接移植** | 本機已有 resident DiT 及 schedule reuse，但沒有該跨程序 cache。來源 checkpoint fingerprint、LoRA 綁定及 AdaLN final dtype 均需本機化。 |
| **A5：PR55 預量化權重串流** | **記憶體備案；拒絕作速度優先方案** | 雙槽逐層讀取 INT8 weights/scales；原版 fingerprint 不一致只警告後繼續。不能保障目前 adapted weights，且會引入每次 forward 的 I/O。 |
| **A6：殘餘十變體 pipeline** | **可優先採用，須等值驗收** | 中央 `h3_gpu_pipeline()` 已 lazy；呼叫端仍在選路前取得十個 variant，並要求全部可用。應只索取實際選中的 FC1／FC2 pipeline，保持原選路與 dispatch。 |

來源：本機 [`h3_gpu.m`][B-GPU]、[`h3_dit.c`][B-DIT]、[`h3_video_encoder.c`][B-ENC]；固定 [PR64][P64]、[PR77][P77]、[PR75][P75]、[Conv3d arena][J-ENC]、[DiT cache][J-CACHE]、[PR55][P55]。

---

## 二、證據、覆蓋範圍與不可變契約

### 2.1 承接的三份文件

已完整讀取：

- `luna-to-astra-portability.md`，包含完整 raw 內容及獨立 raw 第 9 行。
- `h3-quality-safe-contract.md`。
- [`research/eros-beta5-luna-forks-platforms-2026-10-26.md`][LUNA]。

本次 session 的 `local://` 別名沒有對應檔案；已改讀提供文件所在的原 session `local/` 實體檔，未以摘要替代原文。

Luna 的實際 discovery 覆蓋為：

- **47 筆 PR**，不是 100。
- fork API 回傳 **243 個 unique repositories**，metadata 顯示 247；不把差額自行解釋為漏審的四個公開 fork。
- **15 個 fork branch API、5 組 compare histories／檔案列表**。
- 以上不等於逐一審核 243 個 fork、全部分支或全部 diff。

### 2.2 已安裝 baseline 身分

本機 [`runtime-source.json`][B-ID] 記錄：

- source snapshot：`0f96a9bd859f1576097327e2b7f46298b28d31d8`
- executable SHA-256：`e5a88771f646cfc4bcc01b87011255ae10a14c78c47055b92fd37af2ba8689ab`
- 原始模型 SHA-256：`098f138f3e899d03821dd8296c8db3db6fa16371e50fcdd38661d347fd9a1dfa`
- 轉換模型 SHA-256：`a61c17e55237c03f64c3db2620d11a09b5628fdf95fbdf7ac0f78c9dcc4f51bf`

這是已讀的 provenance 記錄，不是本次重新 hash 或重建認證；該檔亦明載 build reproducibility 尚未認證、既有 acceptance 不構成完整聲音／嘴型品質認證。

### 2.3 主程序提供、未重跑的觀察

- macOS **27.0.1／26A434**，128 GiB。
- SDK **27.0**。
- active DeveloperDir 為 `/Library/Developer/CommandLineTools`；`xcodebuild -version` 無法使用。
- 主程序的隔離 GPU 測試能使用 runtime dynamic Metal compilation。
- **不能由上述事實推導 full Xcode GPU capture 工具已安裝。**
- 隔離開發樹的 FC2 K1024 為 default-off；主程序回報 128／129／257／45001 rows byte oracle 通過。
- 固定順序三次 whole-MLP 合計：K512 `0.603250 s`、K1024 `0.590337 s`；ABBA 尚待完成。**這不是穩定加速證據，也不是本報告移植來源的測量。**

### 2.4 不動的品質邊界

維持四步 ER-SDE／beta57、全部 50 layers、reuse/core=1、768×1344、現有 INT8 projections、BF16 video VAE、F32 audio，以及 beta5 **BF16 RMSNorm → F32 final modulation／output heads**。

不以 sparse/local attention、W4A8、Sage／FP8 attention、跨步 KV／residual reuse、減少 references／片長／解析度／layers／tokens 交換速度。

影片程序約 40 GiB 是軟預算；128 GiB 實體記憶體不是可全部交給影片程序的額度。LLM 即使不造成容量不足，也可能競爭 GPU 計算與統一記憶體頻寬。

---

## 三、A0／A1：實際 attention 差異與 rectangular query 移植

### 3.1 本機目前做什麼

`h3_gpu_sdpa_graph()` 的 cache key 包含：

- dtype、batch、sequence、heads、head dimension；
- scale、causal；
- input head-major 與 output head-major。

Q/K/V placeholder 使用相同完整序列 shape；row-major 時先 transpose 成 `[B,H,N,D]`，再呼叫 MPSGraph SDPA。`h3_gpu_sdpa()` 驗證完整輸入／輸出元素數與 dtype，消耗並清除 `headMajorSDPAInputs`，最後把 `command.rootCommandBuffer` 回寫到 GPU context。[B-GPU]

本機 `run_block()` 已串接：

1. `h3_gpu_grouped_qkv_linear_rope_int8()`；
2. `h3_gpu_sdpa_bf16_head_major_output()` 或一般 BF16 SDPA；
3. `h3_gpu_linear_int8_head_major_bf16()` 或 row-major output projection；
4. 已融合的 attention gate／MLP AdaLN／量化。

因此，新增 attention 不應拆掉既有 head-major producer／consumer，也不應重新導入完整 transpose buffer。[B-DIT]

### 3.2 PR64 與 PR77 並不相同

| 項目 | PR64 固定 head | PR77 固定 head |
|---|---|---|
| 預設 query block | 2048 | 0，opt-in |
| split threshold | 12288 | 8192 |
| graph key | 加入原始 query block、threshold | 先算有效 query block，再加入 key |
| Q slicing | `sliceTensor:qt dimension:2` | 相同 |
| K/V | 每塊皆使用完整 `kt/vt` | 相同 |
| 尾塊迴圈 | `start += query_block`，有 `start + rows` 加法 | `rows=MIN(block, sequence-start)`、`start += rows` |
| 輸出 | graph concat，dimension 2 | 相同 |

PR64 在極端 `uint32_t` 範圍有可避免的加法／迴圈溢位風險；PR77 的剩餘長度寫法較適合作移植基礎。這不是說目前約 45k rows 已觸發該風險。[P64][P77]

**重要修正：兩個 PR 都不是把整個 host 執行改成受控雙槽串流。**它們在一個 graph 中建立多個 rectangular SDPA nodes，再 concat。MPSGraph 是否重用 intermediate、是否並行保留多塊結果、concat 是否增加暫存，必須觀察，不能由程式碼中的 `slice` 判定。[P77]

### 3.3 建議的兩層實作邊界

**第一層：最小差異的 graph slicing**

保留完整 placeholder、現有 `h3_gpu_sdpa()` API 及 layout 傳遞。只在 noncausal graph 裡切 Q，使用 PR77 尾塊邏輯，預設關閉。

此版本的價值是：

- 以小變更檢驗 rectangular SDPA 在本機是否正確；
- 避免 host 自行處理跨 head stride；
- 比較完整與切塊 graph 的實際時間及 residency。

不能預先把它標成降記憶體優化。

**第二層：只有第一層顯示必要時，才做獨立 rectangular dispatch**

每次處理 `M=min(block,N-start)` 個 query rows，完整 `N` 個 keys／values；重用明確大小的 Q／O scratch，再 scatter 回完整 output。

此時 graph geometry 至少須區分：

`dtype, B, M, N, H, D, scale, causal, input/output layout`

若 graph 內嵌 slice start，也必須把 start 納入 key；若先 pack 到相同 compact scratch，graph 才能只按完整塊／尾塊 geometry 重用。

### 3.4 stride、cache 與輸出界線

以 head-major `[B,H,N,D]` 為例，query 起點為 `s`、長度為 `M`：

\[
\operatorname{offset}(b,h,r,d)=((bH+h)N+s+r)D+d
\]

head stride 仍是 **`N×D`**，不是 `M×D`。把 pointer 加上 `s×D` 後直接宣告成 contiguous `[B,H,M,D]`，會錯讀第二個 head 起的內容。

可接受方案：

- graph 內的真正 tensor slice；
- 經確認支援、且完整 cache geometry 正確的 strided tensor view；
- 明確 pack／scatter。

最後一個方案增加搬運，但比未驗證的 offset-only view 安全。

輸出亦須保留：

- 完整容量 `B×H×N×D`；
- 每塊只寫 `M` rows；
- 最末塊不寫 padding；
- head-major 與 row-major scatter 各自正確；
- checked-size 乘法及 byte offset；
- `headMajorSDPAInputs` 一次性狀態不可在 tile loop 中被錯誤重解讀。

本機 `h3_gpu_graph_data()` 的快取不能被當成任意 offset／stride view 的證明；若新 view 使用它，必須確認 shape、offset、stride 與底層 buffer 身分的快取語義。[B-GPU]

### 3.5 full-KV 數值門檻

以下為**主程序未來的驗收要求，本次未執行**：

- 小矩陣：全 tensor 對獨立 F64 dense oracle，而非只挑幾個 channels。
- 大矩陣：至少涵蓋 12256、32768、45001 與 `block±1`；使用正式 H=56、D=128。
- 約 45k rows：完整比較 baseline 與 candidate output；獨立高精度 oracle 可採少量 query rows，但每個抽查 row 必須計入 **全部 N 個 K/V**。
- 覆蓋 uniform logits、極尖峰 logits、大幅值、近乎抵消、零值、尾塊、不同 head 的獨特資料。
- 明確檢查首尾 reference／audio／video 區段；只擾動遠端或最後一塊 K/V，也必須能影響應受影響的 query。
- 報告非有限值、max absolute error、RMS／relative RMS、BF16 ULP 分布及不相同元素比例；不能只報平均誤差。
- 驗證輸出後的既有 INT8 attention-output projection：小 attention 差異可能改變 row scale 或量化邊界。
- 若要求 bitwise 路線，任何 byte 差異都不算通過；若改走極輕微 FP 差異路線，門檻須另行批准，且仍不能越過影片品質門檻。

PR77 附帶 sampled probe 的 `<0.001 max error`／`<0.02 relative RMS` 是其 probe 條件，**不是本案可直接採用的身份／聲音品質允收值**。[P77]

---

## 四、A2：macOS 27 TensorOps 與真正 full-KV online softmax

### 4.1 已有 API 證據，尚無本機特定 shader 證據

Apple WWDC26 session 330 明確展示：

- `matmul2d` 計算 QK；
- cooperative tensor 承載中間值；
- `reduce_rows` 計算 row reduction；
- `map_iterator` 對應 row max；
- `get_left_input_cooperative_tensor()` 接到後續 PV matmul；
- `is_compatible_as_left_input`／right-input 類型相容檢查；
- 不相容時仍需經 threadgroup memory 轉換 layout。

Apple 也明確區分：macOS 26 需要先存到 threadgroup memory 的某些接法，macOS 27 可直接用 cooperative tensors 作 matmul input。[APPLE]

因此：

- **可研究**：目前主機 OS／SDK 版本符合研究前提。
- **未證實**：所需 BF16 input、F32 accumulator、reduction layout、第二段 matmul input 的精度／layout 組合。
- **未證實**：register pressure、spill、occupancy、threadgroup memory 及約 45k rows 的效能。
- **未執行**：feature compile、pipeline creation 或 GPU gate。
- full Xcode 缺席不能被誤寫成 runtime Metal 編譯不可用；反過來，runtime 編譯成功也不代表 GPU capture 工具齊備。

### 4.2 必須是 online normalization，不能各塊 softmax 後直接相加

對每個 query row 維持：

- running maximum \(m\)；
- normalizer \(\ell\)；
- 未正規化 weighted-value accumulator \(a\)。

對新 K/V tile 的 scores \(s\)：

\[
m'=\max(m,\max s)
\]

\[
\ell'=e^{m-m'}\ell+\sum_j e^{s_j-m'}
\]

\[
a'=e^{m-m'}a+\sum_j e^{s_j-m'}v_j
\]

全部 K/V tiles 完成後才輸出 \(a'/\ell'\)。

這保留所有 attention edges，避免完整 \(N^2\) score matrix；但浮點加法次序、exp、matmul accumulation 及暫存 rounding 仍可能改變結果。FlashAttention 的「exact attention」指未採 sparse 近似，不代表與 MPSGraph bitwise 相同。[FLASH]

本案建議：

- Q/K/V 維持既有 BF16 邊界；
- row max、normalizer、累積值使用 F32；
- output 在既有邊界寫回 BF16；
- 不為了接 cooperative matmul 而未經評估地把 probability／accumulator 降為 BF16；
- 不改 Q/K norm、partial RoPE、scale、packed row order；
- 不把 INT8 projection 的存在解讀為 attention 量化也「同品質」。

**決策：A2 不排在第一個落地 patch。先取得 A0／A1 的約 45k rows 歸因，再決定是否值得承擔新 reduction kernel 的品質風險。**

---

## 五、為何「逐幀生成完就丟掉」不等於 full dense 串流

ComfyUI `Attention.forward()` 在 packed sequence 上呼叫無 mask 的 `optimized_attention()`；Diffusers `MiniMaxH3AttnProcessor.__call__()` 明確使用 `is_causal=False`；LightX2V `_attention()` 同樣傳入 `causal=False`。[COMFY][DIFF][LIGHT-ATTN]

packed sequence 包含文字、reference／condition rows、audio 與 video。這不是自回歸 LLM 的 causal prefix。

因此：

1. 某一幀的 query 依賴其他時間與其他模態的 K/V。
2. 下一層的 K/V 又依賴上一層完整 attention 更新後的 hidden states。
3. 下一個 ER-SDE transition 還會改變 target latent。
4. reference 原始 latent 固定，不代表 reference rows 在每一層的 hidden／KV 都可跨層、跨步固定重用。

**合法的 streaming 單位是同一層、同一步內的計算／資料 tile，不是把未來 frames 從注意力範圍移除。**

最具體的保守路徑為：

> 同一步、同一層產生完整 K/V → query tiles 逐批讀取全部 K/V → 寫完該層所有 output rows → 才進入依賴它們的後續層。

若按幀分組 query，但每組仍讀完整 K/V，語義可以保留；這只是 query tiling 的一種排序，不是低延遲的獨立逐幀完成。

---

## 六、A3：Conv3d arena 是真增量，但不在去噪熱路徑

### 6.1 本機與來源的差異

本機：

- `run_conv()` 配置 padded／output tensor，開始 command，執行 convolution，再同步 submit。
- `run_block()` 配置 norm1、pad1、hidden、norm2、pad2、output，必要時另配 shortcut。
- `encode_tile()` 逐層接續，釋放上一段 tensor。[B-ENC]

固定來源：

- `prepare_planar_arena()` 走訪 network geometry，配置共用 pixels、兩個 state、norm、padded、hidden、shortcut。
- `planar_conv(..., conv3d)` 的 Conv3d 分支仍呼叫 `h3_gpu_conv3d_f32()`，保留 temporal front padding。
- `planar_block()` 重用上述 scratch。
- `encode_tile_planar()` 以 `h3_gpu_continue()` 提交中間階段，最後 `h3_gpu_submit()` 等待一次 tile。
- `h3_video_encoder_open()`／`encode()`／`free()` 提供可重用 encoder context。[J-ENC]

固定 head 已包含 Conv3d arena 分支，不能把整個來源簡化成「Conv2d 近似」。但其 `planar` 路徑處理的是單幀；多幀 reference 不應直接套用單幀容量公式。

### 6.2 可取與不可取

**可取：**

- 同 geometry 的 scratch reuse；
- 原 Conv3d graph、F32 weights／norm／padding；
- 保持相同 tile 與 stitching；
- GPU 完成後才讓 CPU 重寫 pixels 或重用不再安全的 buffer。

**不直接採用：**

- 切最後 temporal tap 改 Conv2d；
- 改 encoder tile 尺寸；
- 連帶搬入 shared GPU context／提早 DiT load 的整包生命週期變更。

單幀 Conv2d 在實數上可利用其他 temporal taps 乘零的性質，但換 convolution primitive 仍可能改變浮點實作；本案沒有必要把這個風險綁進 allocation patch。

### 6.3 記憶體不能只看 allocation 次數

若 `A` 為最大 activation 元素數、`P` 為最大 padded 元素數、`X` 為 pixels 元素數，來源 arena 的主要 F32 容量為：

\[
4(X+5A+P)\ \text{bytes}
\]

另有 weights、graph workspace、各 tile latent、stitch output 等。

重用 arena 減少配置次數，不保證 peak 比原版低：多個「全網路最大值」buffer 可能同時常駐；若 encoder context 在去噪期間保留，arena 還可能從短暫成本變成持續成本。

驗收應使用合成單幀 pixels，逐 tile latent 與 stitching 結果 byte-compare，覆蓋邊界尺寸、不同 geometry 連續請求及重複使用。這是建議的非影片驗收，本報告未執行。

---

## 七、A4／A5：兩種權重 cache 的移植限制

### 7.1 A4：DiT-load cache 不等於去噪中逐層串流

jerryfane 的 `h3_dit_cache.c` 提供：

- `h3_dit_cache_checkpoint_fingerprint()`；
- `h3_dit_cache_int8_key()`；
- `h3_dit_cache_adaln_key()`；
- header／長度驗證；
- 私有暫存檔、`fsync()`、`rename()` 發布；
- 額外的 recompute／byte verification 模式。[J-CACHE]

其 checkpoint identity 使用 resolved directory、檔名、檔案長度及 mtime；variant 加入 OS build、GPU 描述、shader source、量化選項等。**它不是模型內容 SHA-256，也沒有自動納入本機動態 LoRA stack。**

本機 `quantize_block()` 已將啟用的矩陣集中於一次 begin／submit，完成後才釋放 BF16 source；`h3_dit_prepare_request()` 已可保留 resident weights，並按完整 AV sigma arrays 與 condition presence 保留 schedule。[B-DIT]

因此 A4 的新增價值主要是：

- 新程序重複載入同一有效模型；
- 避免再次執行相同 weight quantization；
- 不是讓每個 denoise block 少做 attention／MLP。

### 7.2 beta5 AdaLN 是直接移植阻擋點

來源 `project_schedule()` 走 full-width time embedding；`schedule_cache_header()` 把 blocks **及 final** 都以 `uint16_t` 大小儲存；`load_cached_schedule()` 亦按 BF16 載入。[J-SCHED]

本機 beta5：

- `validate_curve_schema()` 驗證 `adaln_t_table` 與 F32 rank8 projections；
- `curve_project()` 產生 block modulation，再落到 BF16；
- final modulation 使用 F32 allocation 與 `h3_gpu_linear_f32()`。[B-SCHED]

所以不能僅把 cache functions 接上去。cache schema 至少要綁定：

- full-width／rank8 curve schema；
- table、basis／bias 的有效模型身分；
- interpolation 與 projection 算術版本；
- AV sigma 的原始位元及 condition 模式；
- **block BF16、final F32 的分別 dtype／長度**。

也不能因為它叫 timestep cache，就把它誤判成被禁止的 cross-step target KV reuse；它們是不同資料依賴。

### 7.3 本機已有 LoRA identity，但要保留其真正順序語義

本機 `h3_weight_store_bind_loras()` 要求在任何 payload load 前綁定 stack；weight loader 遇到 target 時走 `h3_lora_load_target()`。[B-WEIGHT]

`h3_lora_stack_load()`：

1. snapshot adapter；
2. 去掉 strength=0 的 inactive entries；
3. 以 `compare_entry()` 依 adapter content hash、strength、path 排序；
4. 按此 canonical order 建立 factors；
5. identity hash 納入 dialect、adapter digest、strength 的 F32 位元。[B-LORA]

**因此，不能要求「只要 CLI 參數換順序就一定 cache miss」。**本機先 canonicalize；應綁定實際 canonical composition order 與其演算法版本，而不是自行改成輸入順序。

必要 cache identity 應組合：

- base model content／conversion identity；
- FL2VA／Ref2VA target 與 tensor layout；
- 現有 `h3_lora_stack_identity()`；
- canonical ordering／composition／rounding 版本；
- quantizer／shader／dtype／矩陣集合；
- 可能影響產物的 OS／device／compiler contract。

不可先量化 base，再用另一條算術路徑「補上 LoRA」，卻稱與現有 adapted-BF16→INT8 相同。

### 7.4 A5：PR55 實際串流的是 weights

固定 PR55：

- builder `quantize_and_write()` 使用 `h3_gpu_quantize_weight_int8()`；
- cache record 包含 QKV／OUT／FC1／FC2 的 INT8 weights 與 F32 row scales；
- `allocate_attention_slot()` 建立雙槽；
- `read_attention_layer()` 讀 QKV／OUT，選配 FC1／FC2；
- `attention_cache_offsets()` 決定逐層檔案位置。[P55][P55-BUILD]

這可以保留原 attention connectivity，但必須確保 cache bytes 就是目前有效 adapted weights 的量化結果。

**原版不可直接接受的細節：**

- `attention_cache_validate()` 遇到非零 fingerprint 不符，輸出 warning 後仍 `return 1`。
- unknown model kind 會跳過該項匹配。
- builder 沒有綁定本機 LoRA stack 的流程。
- builder 直接寫 output，不能沿用為本機可靠發布契約。
- 本機已有「active LoRA 不可用既有 SSD streaming」的保護；不能繞過該限制就稱支援完成。[P55][P55-BUILD][B-CORE]

本案應把不符 cache 視為 miss 或明確失敗；不得只警告後使用可能不同的角色／聲音權重。

---

## 八、約 45k tokens 的記憶體帳，不把配置量當 peak

以下為 **N=45001、B=1、H=56、D=128、hidden=5376、FFN=14336** 的形狀推導，並非五張 reference 的實測 production N。實際 N 必須從 packed layout 取得。

### 8.1 主要 activation buffers

| Buffer | 元素／格式 | 推導容量 |
|---|---:|---:|
| Q、K、V，各一份 | `N×7168` BF16 | 各約 **0.601 GiB** |
| 完整 K+V | 同上兩份 | 約 **1.202 GiB** |
| `qkv` arena | `N×7168×3` BF16 | 約 **1.802 GiB** |
| `hidden` | `N×5376` BF16 | 約 **0.451 GiB** |
| `mod_attention` | 同上 | 約 **0.451 GiB** |
| `attention_output` | 同上 | 約 **0.451 GiB** |
| `activated` | `N×14336` BF16 | 約 **1.202 GiB** |
| `int8_activation` | `45056×14336` INT8 | 約 **0.602 GiB** |
| grouped activation scales | `45056×14` F32 | 約 **2.406 MiB** |

依本機 `allocate_activations()`，開啟 aliases 時：

- `attention_heads` 與 `mod_mlp` alias `qkv`；
- `mlp_output` 在執行時使用 `attention_output`；
- 不能再把這些 alias 全額加一次。[B-DIT]

上表主要獨立 allocation 合計約 **6.76 GiB**。尚未包括 RoPE、row maps、input／final tensors、refiner、scheduler state、MPSGraph workspace、weights、其他模型或 allocator 保留；也不是所有頁面必然同時 physical-resident。

另有一個可見機會：直接 INT8 QKV 路徑不需要完整三倍寬的中間 QKV，但 baseline `qkv` arena 仍配置三倍寬。縮成後續 aliases 所需最大容量，在此例理論上可少約 **1.20 GiB 的 allocation**。這需要證明所有 BF16 fallback／diagnostic 路徑、初始化時機與 alias 使用範圍；不能只改一行容量，更不能直接稱 physical peak 降低 1.20 GiB。

### 8.2 query block 並不自動消除大型 scores

若某 backend 真的 materialize scores：

| 假設 score tensor | BF16 容量 | F32 容量 |
|---|---:|---:|
| 完整 `[56,45001,45001]` | 約 **211.2 GiB** | 約 **422.5 GiB** |
| query block 2048 的 `[56,2048,45001]` | 約 **9.61 GiB** | 約 **19.23 GiB** |

這些是形狀推導，**不是宣稱目前 MPSGraph 配置了這些 tensors**。目前 backend 可能使用融合／分塊實作，內部尚未查明。

N=45001、block=2048 有 22 塊，末塊 1993 rows。獨立 compact Q／O scratch 各約 28 MiB，合計 56 MiB；但這不包含 SDPA 內部 workspace。

若每個 query tile 都從記憶體完整重讀 K/V，22 次的邏輯讀取約 26.4 GiB／layer／forward；實際 DRAM traffic 受 cache、head scheduling 與 kernel tiling 影響。切塊不是免費的記憶體最佳化。

### 8.3 權重與 PR55 雙槽

依四個主矩陣尺寸：

- QKV＋OUT INT8 weights：約 **147 MiB／layer**。
- FC1＋FC2：約 **220.5 MiB／layer**。
- 四個矩陣加 F32 scales：約 **367.73 MiB／layer**。
- 50 layers：約 **17.96 GiB**，不含其他權重。
- 四矩陣雙槽：約 **0.718 GiB**，不含 activation、norm、其他權重與 file cache。

若四步各走一次完整 50-layer forward，完整權重串流的邏輯讀量約：

\[
4\times17.96 \approx 71.8\ \text{GiB}
\]

若有效讀取吞吐為 \(B_{\mathrm{eff}}\)，純傳輸時間的理論下限是：

\[
T_{\mathrm{I/O}}\geq \frac{71.8\ \text{GiB}}{B_{\mathrm{eff}}}
\]

可以和 GPU 計算重疊，但不能假設全部隱藏；page-cache hit 也不等於沒有 memory traffic。這正是 A5 在 128 GiB 主機上應先視為容量備案，而非加速方案的原因。

### 8.4 peak 驗收要分開記帳

至少區分：

1. 活著的 Metal buffer bytes；
2. mmap virtual bytes；
3. 檔案頁面 resident／wired 狀態；
4. MPSGraph／driver workspace；
5. encoder／DiT／decoder 同時存活的 overlap；
6. 程序 physical footprint；
7. 全機 memory pressure、swap 與 LLM 工作負載。

A4 的「zero-copy cache mapping」不是零 residency；來源還會在 load 階段觸碰 cache tensors 使其 resident。early load 若與 text／reference encoder 重疊，也可能增加 peak。[J-SCHED]

---

## 九、A6：只修呼叫端，不重做 PR75

本機 `h3_gpu_mlp_int8_bf16()` 在計算選路 predicates 前，依次呼叫中央 getter 取得：

- FC1：generic、`k5376`、full-`k5376`、local；
- FC2：generic、full-`k14336`、`r128x256`；
- grouped FC2：一般、local、local128。

隨後要求十個 pipeline 全部存在並符合 threadgroup 限制，才計算：

- `int8_fc1/2`；
- known／full／local FC1；
- grouped／row FC2；
- local128 與 rows≤2048 的 row-FC2 variant。[B-GPU]

**建議變更：**

1. 原樣計算 predicates。
2. 按目前同一優先順序決定 FC1、FC2 名稱。
3. 只對實際啟用的 stage 呼叫 `h3_gpu_pipeline()`。
4. 只驗證選中 pipeline 的 threadgroup 能力。
5. 完全保留 quantizer、clip、K 分組、buffer binding、dispatch geometry 與 BF16 邊界。

原中央 getter 的成功快取、library compilation 時機均不變。[P75][B-GPU]

風險是選路 precedence 被「順便簡化」而改變；尤其 grouped local128、grouped local、row-FC2 邊界不能重排。驗收要包括目前支援的診斷 stage 與 flag 組合。

效益分類是**首次進入該呼叫的 pipeline preparation**。warm denoise 不會每次重新編譯這十個 pipeline，因此不能宣稱每層省十次 compile。

---

## 十、建議的四個增量 patch 與驗收順序

排序依「可證明性、改動範圍、對目前需求的價值」，不是未測量的速度排名。

### Patch 1：A6，selected-only pipeline lookup

- **檔案／介面：**`h3_gpu.m::h3_gpu_mlp_int8_bf16()`；不變更 exported API。
- **數值門檻：**實際選中 kernel、args、dispatch 與原版一致；activated、量化資料／scales、最終輸出 byte-identical。
- **案例：**128／129／257／2048 邊界及45001 rows；既有合法 diagnostics。
- **效能門檻：**分開測 fresh-process library compile、首次 MLP、warm MLP。只報所省 pipeline 建立時間；warm 不得出現可重現退化。

### Patch 2：A1，default-off full-KV query slicing

- **檔案／介面：**`h3_gpu_sdpa_graph()`／`h3_gpu_sdpa()` 內部；保留現有 BF16 head-major input/output API。
- **第一階段：**PR77 式 graph slice＋concat；不新增多種串流架構。
- **數值門檻：**第三節完整 gate；保證所有 K/V、row order、layout、尾塊及 cache identity。
- **效能門檻：**約45k rows 的完整 SDPA與既有 output projection；同時記錄 physical footprint、buffer residency及完成後 wall time。
- **升級條件：**只有證據顯示 graph intermediate residency／concat 是問題，才改獨立 rectangular dispatch。

### Patch 3：A3，單幀 Conv3d scratch arena

- **檔案／介面：**`h3_video_encoder.c` 的現有 encoder context、`run_conv()`／`run_block()`／`encode_tile()`；維持外部語義。
- **範圍：**只取 allocation reuse，保留 Conv3d；將 continue／wait 改動與 arena 等值驗證分開觀察。
- **數值門檻：**相同 pixels、weights、tile geometry 下的完整 latent bytes／stitch output 相同；重用前必須證明上一個使用者已完成。
- **效能門檻：**單張與多張依次編碼，分 cold context／warm context；報 allocation 次數、wait／wall time、arena peak。
- **邊界：**不能把結果算進 DiT denoise 加速。

### Patch 4：A4，僅持久化已適配的 INT8 block weights

- **檔案／介面：**接在本機 `load_core()`／`quantize_block()` 與已綁 LoRA 的 weight loader；不套用來源整包 early-load。
- **第一個可控範圍：**只 cache 四個主矩陣與 scales；**先不移植 AdaLN persistent cache**，避免把 final BF16 schema 帶入 beta5。
- **數值門檻：**hit bytes 必須等於相同有效模型現算的 quantizer bytes；模型、strength、canonical composition、target、shader／版本改變必須失效。
- **檔案門檻：**截斷、錯長度、錯 dtype／schema、不完整發布不可使用；內容校驗與 provenance 不能只靠 mtime。
- **效能門檻：**分 cold filesystem、warm file cache、新程序及既有 resident request。若實際工作流保持程序常駐、沒有重複載入痛點，則不值得優先落地。

A2 保留為後續研究；A5 保留為超出記憶體預算時的獨立備案，不與上述 patch 綁包。

### 共通效能判定

由主程序在隔離環境執行：

- 交錯 ABBA 或平衡順序，不能只做全部 A 再全部 B；
- 準備／compile 與 warm execution 分開；
- 等待完整依賴完成後量 wall time；
- 不用 root GPU counter 單獨代替完成時間；
- 記錄並區分 LLM contention、熱狀態及記憶體壓力；
- 包含45001 rows，不以2048 microcase代表 production；
- 結果小於 run-to-run variation 時，結論是「未建立收益」。

不設定沒有依據的加速目標。若某區段占總時間 \(f\)，且該區段未來實測加速 \(s\)，整體理想上限才可用：

\[
S=\frac{1}{(1-f)+f/s}
\]

目前尚無本案 A1／A2 的可信 \(f\) 與 \(s\)，不代入猜測值。

---

## 十一、平台借鏡：只取語義與可移植設計

| 來源 | 已讀的具體 seam | 可借鏡／不可直接搬入 |
|---|---|---|
| **ComfyUI** | `Attention.forward()`、Q/K norm＋partial RoPE、無 mask packed attention；`AdalnProj` 所在模型檔 | 確認 dense joint AV/ref 語義。VSA gate 的存在不代表 dense forward 可以省略 edges。GPL 程式不能當 MIT snippet 貼入。 |
| **LightX2V** | `_prepare_qkv()`、`_attention()`、`infer_block()`、`_compute_adaln_table()`；`adaln_cache.py::_projection_contract()`、timestep bit keys | 可借鏡 cache 對 precision/backend 的明確綁定與 full-sequence attention seam。CUDA／Triton／SGL／FP8、SP/TP 並非單 Apple GPU 的現成加速。 |
| **Diffusers** | `MiniMaxH3AttnProcessor.__call__()`、`MiniMaxH3AdaLayerNormModulation.forward()`、`MiniMaxH3AdaLayerNormOut.forward()` | 確認 full packed self-attention、`is_causal=False`、modality indexing 與 dtype 邊界的重要性。該 full-width AdaLN 不能取代本機 beta5 rank8 table。 |
| **FlashAttention 論文** | IO-aware exact dense attention | 可獨立實作 online-softmax 演算法；CUDA kernel 與其 benchmark 不可直接轉成 Metal 宣稱。 |
| **Apple TensorOps** | cooperative tensor／row reduction／matmul compatibility | 是 native kernel 研究依據；不是目前 MPSGraph 內部實作證據。 |

來源：[COMFY]、[LIGHT-ATTN]、[LIGHT-CACHE]、[DIFF]、[FLASH]、[APPLE]。

ANE 分支使用其他 execution／precision contract；VDN 分支改模型架構與 temporal state；CUDA 分支依賴 CUDA libraries。這些不是同模型、同品質的 Metal drop-in。沒有必要為完成本案而擴大成全部 backend 的原始碼審核。

---

## 十二、授權與來源複製限制

### 12.1 程式碼

- 固定 PR55／64／77、jerryfane 與 PR75 的 h3.c 來源採 **MIT**；複製須保留 copyright／license。
- 本機 [`THIRD_PARTY_NOTICES.md`][B-NOTICES] 另記錄 ccv 衍生 Morton／TensorOps 設計的 **BSD-3-Clause** attribution；不能只保留 repository 根目錄 MIT。
- **ComfyUI：GPL-3.0。**可作語義研究；若複製／衍生其程式，必須處理 GPL 義務，不得宣稱已自動轉為 MIT。
- **LightX2V／Diffusers：Apache-2.0。**複製時保留適用 notice／license、標示修改並遵守相關條款；CUDA／Triton／SGL 等外部依賴另查。
- **Apple 文件／範例：Apple 條款。**API 與演算法可供研究，不代表投影片 snippet 自動具 MIT 授權；本次未完成下載範例 package 的逐檔授權審核。
- FlashAttention／PyTorch 的演算法或文件引用，不等於已批准複製任意 upstream kernel；選定檔案／revision 的 license 須另核。
- 本機 notices 已指出 RES4LYF 固定來源有 AGPLv3 加額外商業服務限制；本案不移植 sampler 程式，維持既有 ER-SDE。[B-NOTICES]

這是工程複製限制整理，不是法律清關。

### 12.2 模型

h3.c 的 MIT **不授權模型權重**。Eros 模型卡指向 MiniMax H3 Community License，且作者說明混合來源可能承接其他模型社群授權。[EROS][MODEL-LICENSE]

因此，衍生量化 cache、轉換權重、LoRA 合成權重及其散布，仍需檢查模型與 adapter 的適用條款。不能因為 cache 由 MIT builder 產生就視為 MIT 資料。

Eros `main` 模型卡是可變來源；本機實際安裝 beta5 的模型 SHA 與 conversion provenance，應優先於後續模型卡對 beta6 或其他變體的說明。[B-ID][B-SETUP]

---

## 十三、實際 primary-source read ledger 與限制

| 來源 | 固定版本／定位 | 實際閱讀範圍與用途 |
|---|---|---|
| 已安裝 runtime | snapshot 記錄 `0f96a9b…`；唯讀 SSH | `h3_gpu.m` SDPA graph／dispatch、MLP pipeline、command timing；`h3_dit.c` allocation、quantize、resident prepare、run block；encoder allocation；schedule rank8／final；LoRA identity／binding；provenance／notices。不是全 repository audit。 |
| PR64 | `39e95698543b9d78ad0152a61c2b36b43cc4e25c` | 完整相關 patch：query defaults、key、slice、loop、CLI。 |
| PR77 | `0a95ea10c0cab51d16ef094ee88dba898e23524d` | 固定 patch與 SDPA implementation；包含 probe 條件的閱讀，未執行 probe。 |
| PR75 | `30b63b51404cae1f2dcc1841abea96f1893c5b49` | 中央 getter、create 流程及固定 provenance；與本機十變體 caller 比較。 |
| jerryfane encoder | `3d0e90304639a61455a955b4ae14884f6d2dfdec` | arena sizing、Conv3d／Conv2d 分支、block／tile execution、encoder lifetime；不是整個 integration 分支審核。 |
| jerryfane cache | `420df19b3dc3fd4b406c0df83ba6911bd756b31e` | fingerprint／key／header／publication、AdaLN serialize／load／verify、相關 allocation 片段。 |
| PR55 | `8b51e19b9b88ce09e670039aebf8b7e98697e785` | builder、record layout、validation、slots、讀取函式與量化函式。 |
| ComfyUI | `cdf272f3514353aab04dc2d9fceef57a260dc156` | MiniMax model 的 packed attention／norm／RoPE seam、license。 |
| LightX2V | `100ecfec4be2956539f41688cbd7bc96f0528cca` | H3 attention／AdaLN inference、cache contract、license。 |
| Diffusers | `3b56af0c5268049588cf470e38955f4c18002498` | H3 attention processor、modulation／final norm，檔頭 Apache notice。 |
| Apple | WWDC26 session330 | transcript／API examples；不是本機 feature compile 結果。 |
| FlashAttention | arXiv:2205.14135 | primary paper；未將 CUDA 實作當已驗證 Metal kernel。 |
| 模型授權 | Eros／MiniMax H3 官方頁面 | 動態模型卡、community license，與本機固定 provenance 分開處理。 |

### 未完成、不能聲稱完成的事項

- 沒有逐一審核所有 fork、全部 PR 或全部 integration diff。
- 沒有讀取正在修改的隔離開發測試檔。
- 沒有編譯 cooperative TensorOps feature probe。
- 沒有本機 A1／A2／A3／A4／A5 新 benchmark。
- 沒有證明 MPSGraph 內部是否 materialize scores 或 child GPU timeline 覆蓋。
- 沒有量到五張 reference 的實際 production sequence N。
- 沒有執行推論、影片生成、測試、build、安裝、下載模型、設定／服務變更或額外模型呼叫。
- 沒有修改或寫入檔案。

**最終判定：先做 A6 的等算術小修；去噪研究集中 A1 的 full-KV query tiling。A3 是獨立參考編碼優化，A4 是有條件的 cold-start 優化，A5 是容量備案，A2 是尚需編譯／數值／效能／品質四道門檻的新 kernel。影片暫停期間，任何 FP reduction 改動都只能取得數值候選資格，不能取得身份、動作、中文聲音、台詞、嘴型與 AV 品質認證。**

---

## 引用

[LUNA]: eros-beta5-luna-forks-platforms-2026-10-26.md
[B-ID]: ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/runtime-source.json
[B-GPU]: ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_gpu.m
[B-DIT]: ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_dit.c
[B-ENC]: ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_video_encoder.c
[B-SCHED]: ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_dit_schedule.c
[B-WEIGHT]: ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_weights.c
[B-LORA]: ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_lora.c
[B-CORE]: ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3.c
[B-NOTICES]: ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/THIRD_PARTY_NOTICES.md
[B-SETUP]: ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/EROS_BETA5_SETUP.md
[P64]: https://github.com/quantz8a/h3.c/commit/39e95698543b9d78ad0152a61c2b36b43cc4e25c
[P77]: https://github.com/flysheep-ai/h3.c/blob/0a95ea10c0cab51d16ef094ee88dba898e23524d/h3_gpu.m#L1462-L1560
[P75]: https://github.com/jerryfane/h3.c/blob/30b63b51404cae1f2dcc1841abea96f1893c5b49/h3_gpu.m
[J-ENC]: https://github.com/jerryfane/h3.c/blob/3d0e90304639a61455a955b4ae14884f6d2dfdec/h3_video_encoder.c
[J-CACHE]: https://github.com/jerryfane/h3.c/blob/420df19b3dc3fd4b406c0df83ba6911bd756b31e/h3_dit_cache.c
[J-SCHED]: https://github.com/jerryfane/h3.c/blob/420df19b3dc3fd4b406c0df83ba6911bd756b31e/h3_dit_schedule.c
[P55]: https://github.com/JunzoKamahara/h3.c/blob/8b51e19b9b88ce09e670039aebf8b7e98697e785/h3_dit.c
[P55-BUILD]: https://github.com/JunzoKamahara/h3.c/blob/8b51e19b9b88ce09e670039aebf8b7e98697e785/h3_build_attention_cache.c
[COMFY]: https://github.com/Comfy-Org/ComfyUI/blob/cdf272f3514353aab04dc2d9fceef57a260dc156/comfy/ldm/minimax/model.py
[LIGHT-ATTN]: https://github.com/ModelTC/LightX2V/blob/100ecfec4be2956539f41688cbd7bc96f0528cca/lightx2v/models/networks/minimax_h3/infer/transformer_infer.py
[LIGHT-CACHE]: https://github.com/ModelTC/LightX2V/blob/100ecfec4be2956539f41688cbd7bc96f0528cca/lightx2v/models/networks/minimax_h3/adaln_cache.py
[DIFF]: https://github.com/huggingface/diffusers/blob/3b56af0c5268049588cf470e38955f4c18002498/src/diffusers/models/transformers/transformer_minimax_h3.py
[APPLE]: https://developer.apple.com/videos/play/wwdc2026/330/
[FLASH]: https://arxiv.org/abs/2205.14135
[EROS]: https://huggingface.co/TenStrip/10Eros-Max/blob/main/README.md
[MODEL-LICENSE]: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE

---

## 後續整合補記：主程序隔離測試結果（非第二次 Astra 回覆）

以下由主程序在上述實際 Astra 諮詢完成後提供；保留前文原始回覆不改寫，亦未再次呼叫模型。本 worker 未執行測試、影片或推論。前文 K1024／ABBA 待完成與 query slicing 待測的文字是諮詢當時快照，現由下列結果更新。

- **FC2 K1024：不採用。**45001 rows、default quant 的 ABBA 12 samples：0.203356 → 0.203962 秒，約慢 0.30%；slow quant 約快 0.28%，屬雜訊範圍。原始來源完整 byte oracle 通過；實驗已封存，active isolated source 已還原。
- **PR77 式 dense query slice／full-KV：沒有建立穩定實用收益，不作預設採用。**H=56、D=128、N=45001，balanced 4 samples：Q1024 為 1.500373 → 1.488571 秒（ratio 0.992134），FP 差異 RMS 0.000792734；Q2048 為 1.483163 → 1.634921 秒（ratio 1.102321），本 fixture bytes 相同；Q4096 為 1.586599 → 1.583225 秒（ratio 0.997873），本 fixture bytes 相同。N=8193、Q1024 ratio 1.156087。不能將單一 case 的微幅改善泛化成加速。
- **記憶體觀察範圍：**long memory guard 以 0.05 秒間隔採樣，peak 6.776 GiB；沒有載入完整權重、沒有影片，不能當作正式影片 peak 或全工作負載記憶體認證。
- 全部 dense candidates 已拒絕預設採用並封存 tar；主程序回報 active isolated `h3_gpu.m`／`Makefile` hash 與 baseline 相同，正式 baseline 未變更。

主程序測試紀錄：[`eros-beta5-quality-safe-fc2-validation-2026-10-26.md`](eros-beta5-quality-safe-fc2-validation-2026-10-26.md)、[`eros-beta5-quality-safe-dense-validation-2026-10-26.md`](eros-beta5-quality-safe-dense-validation-2026-10-26.md)。本補記僅轉述主程序觀察，未重跑或另行品質認證。

因此，前文 A1 是未經當時量測的研究排序，**不是目前的部署／速度推薦**。現有 query slice 實驗未建立收益；未來只有新的具體實作差異與證據才值得重新評估。A6、A3、A4 仍各有前文列明的等值、記憶體及 cold-start 門檻，不因這次 kernel 結果自動取得採用資格。所有身份、動作、中文聲音、台詞、嘴型與 AV 無可察覺退化要求持續有效。

---

## 實際 Astra 第二次諮詢：authenticated Luna 新來源增量

主程序因實質新來源，明確更新原單次諮詢限制並授權一次 delta-only 實際 Astra 諮詢；總計 actual `astra_consult` 呼叫為 **2**。以下完整保留第二次實際回覆，不將 Luna metadata discovery 冒稱為 Astra 全來源審閱。第一份回覆仍原樣保留；前段單次計數是當時快照，現由此更新。

# Astra #2：固定來源增量評估

**結論：新增可研究的價值在 VAE 解碼尾段與計時歸因，不在已被拒絕的 dense attention／FC2 候選。`b0966252` 不宜整包移植；應拆成等算術 epilogue、雙槽排程、最終影格交付三項，各自驗收。`4aed086c` 只取診斷設計，不變更 queue 預設。**

本次判斷屬於**實際 Astra #2**；新增 authenticated metadata discovery 歸屬於**實際 Luna #2 與主程序蒐集流程**。沒有呼叫其他模型／代理，沒有執行命令、build、測試、benchmark、推論、影片生成或檔案寫入。

## 1. 相對 Astra #1 的決策變更

### 覆蓋與主程序結果更新

原匿名 quota 缺口已由 Luna #2 補充資料解決：243 個回傳 fork 的 branch endpoints 全數嘗試，240 成功、345 refs、134 distinct heads；133 個非 baseline heads 全部比較，123 ahead／2 behind／8 diverged，1338 distinct commits，零 compare／commit-pagination errors。保留三個明確 404，以及一個 head 的 300-file 清單截斷；**這不是全面 source、license 或效能審核**。[L2]

主程序的完成結果取代 Astra #1 的待測排序：

- **FC2 K1024：維持拒絕。**default-quant ABBA 12 samples 為 `0.203356 → 0.203962 s`，慢約 0.30%；slow-quant 的 −0.28% 不構成收益。
- **PR77 full-KV query slicing：維持拒絕預設採用。**N=45001 的 Q1024／2048／4096 ratios 分別為 `0.992134／1.102321／0.997873`；Q1024 有 FP 差異，另外兩個僅此 fixture bytes 相同。N=8193、Q1024 ratio `1.156087`。沒有穩定實用收益。
- `6.776 GiB` 是無完整模型權重、無影片的隔離工作負載，以 0.05 秒取樣的觀察，不是 production peak。
- 封存、還原及 active `h3_gpu.m`／`Makefile` hash 相符，均為**主程序回報**；本次未重新 hash 或重跑。[A1]

### 每個新來源的處置

| 固定來源 | 本次 source-level 判斷 | 處置 |
|---|---|---|
| Bambushu [`b0966252`][V] | VAE fused epilogues、雙 latent／projected 槽、逐 chunk assembly、FFmpeg feeder 是實際新增；但含本機沒有的前置 API、生命週期與 unpack 語義差異 | **拆分研究；不整包採用，不接受未驗證 bitidentity 宣稱** |
| Bambushu [`4aed086c`][D] | 新增 MPS root 換手次數及 encode／command-create 時間；未取得完整 child GPU timeline | **診斷取用候選；不是速度 patch，不改 queue 預設** |
| danmo [`02dd9c72`][DN] | 真正 cooperative TensorOps dense kernel；作者 dense 數據輸給 MPSGraph，sparse 路徑另改 connectivity | **僅架構／反例參考；不列移植優先項** |
| qpzmV [`a13b938c`][QP] | 此固定版本實際是 per-lane scalar online-softmax，沒有 matmul2d；且拒絕 head-major | **不移植；後續分支勝出未獲證實** |
| neokree [`23fa7273`][NQ]、[`90624ca4`][NO]、[`d788ad2d`][NM] | 與 PR64／77 同為 full-KV，但不是同一實作：host loop、多次 rectangular graph encode、offset views；oracle 有可取的 production-N 邊界設計 | **只取驗證方法，不重提為已證實加速** |
| Bambushu [`97bca6e9`][RV] | `h3_gpu_sdpa_query_block()` 預設由 2048 改 0；作者註解明言更慢、更多記憶體 | **支持維持現有預設；不是新效能成果** |

既有四步 ER-SDE／beta57、50 layers、core-reuse=1、768×1344、INT8 projections、BF16 video VAE、F32 audio，以及 **beta5 BF16 RMSNorm → F32 final** 全部不變。既定排除的 sparse/local、量化 attention、跨步 reuse 與任何層數／token／reference／解析度／片長削減，不重新開放。

---

## 2. `b0966252`：有真增量，但「同算式」不等於已證明本機 bitwise

### 2.1 與安裝版重疊及移植邊界

安裝版 `h3_video_vae.c::project()`／`run_block()` 已有：

1. F32 norm；
2. cast 至 BF16；
3. BF16 大矩陣投影；
4. **投影結果先落 BF16，再 widen 並加 F32 bias**；
5. F32 QK norm／RoPE／SwiGLU／residual；
6. BF16 attention。

因此，新來源的價值不是再次導入 BF16 VAE，也不是泛稱「改用 TensorOps」。真正差異是 `run_block_fused()` 將這些既有邊界間的 dispatch／中間讀寫合併。[BV][V]

另有兩個直接移植問題：

- 來源呼叫 `h3_gpu_tensor_contents()` 取得 shared-storage view；**本機 `h3_gpu.h/.m` 沒有這個 API**，只有複製式 `h3_gpu_tensor_read_f32*()`。該 commit 依賴其父版本已有的介面，不能視為獨立可套用 patch。
- 本機一般單次生成沒有因這個來源而自動成為 resident-decoder 路徑：`h3_generate()` 目前在 cached／preview decoder 存在時才用 `h3_video_vae_decoder_decode()`，否則走 `h3_video_vae_decode()`／`decode_chunked()`。只修改 resident 函式，不能宣稱一般 CLI 已受益。[BC][BV]

不應為了觸發新路徑，順帶啟用 VAE prefetch、weight cache 或改變 prepared DiT 的保留策略。

### 2.2 各 epilogue 的精度契約

| 新函式 | 保留的必要邊界 | 靜態判斷與風險 |
|---|---|---|
| `h3_gpu_rms_norm_f32_to_bf16()` | 原 F32 `fma` 累積、同 reduction tree、`rsqrt`、兩次乘法，最後才做原 BF16 rounding | host 明確比較 reference／candidate 可用 thread 數，避免 reduction tree 改變；是最容易先隔離驗證的一項 |
| `h3_gpu_video_qkv_bias_rope_bf16()` | BF16 matmul output → F32 bias → F32 QK norm → partial RoPE → BF16 Q/K/V | 保留每 head 的 `[Q,K,V]` 排列、D=64、RoPE half=24；不能換成 DiT 的布局或 norm 路徑 |
| `h3_gpu_bias_swiglu_bf16()` | BF16 FC1 output → F32 gate/up bias → `gate/(1+exp(-gate))*up` → BF16 | 不能把 bias 併入 matmul、提前把 activation 降 BF16，或改 activation 公式 |
| `h3_gpu_bias_scale_add_bf16()` | BF16 branch → F32 bias → 原 scale-add，residual/output 留 F32 | 原本跨 kernel 的 F32 store/load 邊界消失；必須驗證 contraction／rounding 沒變 |

來源與本機皆使用 `MTLMathModeSafe`，BF16 helper 亦採同一 ties-to-even 位元寫法；這是良好的起點，**不是跨 kernel fusion 的 bitwise 證明**。尤其 bias 後的 F32 中間值原來經過記憶體落點；不能僅憑區域變數叫 `float` 就宣告所有 compiler lowering 相同。[V][BG][BS]

**QKV fusion 還有實際成本交換：**來源每個 output dimension 都重算整個 head 的 Q/K norm，並在內迴圈重做 bias addition。本機原 QK norm 也重複讀取，但 bias 原先只在獨立 kernel 計算一次。新增 fusion 減少 traffic／dispatch，同時增加重複 bias 算術；不能由 dispatch 減少推導速度提升。

上述 API 限於 **video VAE block**。不得套到 beta5 DiT final，將其 F32 modulation／heads 降回 BF16。

### 2.3 找到一項明確的本機語義差異

來源 `unpack_patch_rows()` 使用：

```c
value = fminf(fmaxf(value, 0.0f), 1.0f);
```

本機 `unpack_frame_range()` 使用：

```c
if (value < 0.0f) value = 0.0f;
if (value > 1.0f) value = 1.0f;
```

對 NaN 的行為不同；signed-zero 亦不能直接當作 byte-identical。這不是 `b0966252` 新加的 clamp 政策，而是來源分支已帶入、在函式抽取時一起出現的差異。[VS][BV]

**決策：移植 assembly 時保留本機 unpack 運算，不夾帶非有限值政策變更。**作者「與 released path bit-identical」的比較對象是其來源樹，不是本機 baseline。若數值驗證產生 NaN，應判失敗，不能讓來源 clamp 把它掩成合法像素。

---

## 3. 雙槽、stitch、FFmpeg 與影音尾段

### 3.1 雙槽實際重疊的是 GPU 與 CPU，不是兩套 VAE 同時推論

來源的 `decoder_decode_pipelined()`：

- `job = chunk * tile_count + tile`，依 raster order 編碼；
- `slot = job & 1`；
- 先提交 tile N+1，再 `h3_gpu_wait_oldest()` 等 tile N；
- CPU 讀 N 的 projected slot、unpack／stitch 時，GPU 繼續 N+1；
- 下一輪重用槽前，該槽的 GPU producer 已完成，CPU consumer 也已返回。

**只有 latent inputs／projected outputs 雙槽。**weights、hidden、norm、QKV、MLP scratch 仍共用一套，依同一 ordered queue 串行使用。改成兩條 queue 或兩個 concurrent GPU tiles，會破壞這個 scratch ownership 前提，不能照搬。[V]

`h3_gpu_continue()` 追蹤的是 tile 尾端 command buffer；`wait_oldest()` 等其完成後才讓 CPU 看輸出。MPSGraph 自行提交的中間 buffers 不逐一列入 `inflightCommands`，因此：

- **完成性門檻**：tile 尾端完成必須涵蓋其所有 producer dependencies。
- **錯誤門檻**：不能把「最後 root 沒 error」直接當成所有內部 MPS 工作均成功。
- **drain 門檻**：來源 `drain_gpu_chain()` 在 submit 失敗後最多再 wait 三次。這依賴當前 bounded-loop 的 tracked-buffer 上限；不是可任意擴大 queue 深度的通用 drain。
- **取消門檻**：sink 可回傳 0，但來源 `h3_deliver_decoded_frames()` 本身只處理 range／conversion，未檢查 generation cancellation；不能宣稱已有即時取消。必須在槽生命週期安全點停收新工作，退休已提交工作後才能釋放 decoder／共享 buffer。

這些是採用前的 ownership gate；本次沒有執行 fault injection，沒有斷言正常路徑已出現 race。

### 3.2 spatial／temporal assembly 保留了哪些順序

`stitch_tile()` 從原 `stitch_tiles()` 抽出單 tile 運算：

- 上方、左方必須是**同一 chunk**已完成的 tiles；
- 先 top blend，再 left blend；
- 使用原 overlap／keep extent，不改 tile 尺寸或 halo；
- 來源仍保留整個 chunk 的所有 tile RGB，並未縮成單列 halo cache。

`assembler_take()` 在 chunk 最後一個 tile 完成後：

1. 對 chunk>0 的前五幀做原 temporal blend，`alpha=frame/5`；
2. 發送 `[chunk*17, chunk*17+17)`；
3. 保存後五幀；
4. `assembler_finish()` 最後再發送五幀。

總數仍為：

\[
F=17\frac{T-2}{5}+5
\]

適用 `T≥7` 且 `(T−2)%5=0`。例如 T=42 對應 141 幀，最後五幀不可漏送、重送或補成 17 幀；兩-token／五幀診斷路徑仍須留在原實作。[V][BV]

**這是 final latent 已完成之後的 VAE／輸出串流，與 noncausal DiT 的逐幀生成不是同一件事。**沒有放寬任何 packed AV／reference attention dependency。

### 3.3 FFmpeg：有提前交付，但不是 bounded-memory ring

`h3_ffmpeg_av_stream_open()`／`video_feed_thread()` 的所有權設計是：

- 呼叫端先配置**整段 RGB24 buffer**；
- `ready(n)` 在 mutex 下發布 `[0,n)`；
- feeder 在鎖外寫 pipe，成功後才更新 `sent`；
- published RGB 不可再寫，buffer 必須活到 finish／abort 完成；
- `h3_generate()` 的 cleanup 在 `free(rgb8)` 前 abort／join，這一順序正確。[V]

**背壓只會阻塞 feeder 的 `write()`；不會直接阻止 VAE 繼續產生後續影格。**積壓落在整段 RGB24 buffer。因此它減少 whole-video F32 保留、可重疊 encoding，但不能稱為 O(1) 影格記憶體。

其他採用前條件：

- `ready()` 沒有驗證 caller 是否真的填滿前綴；來源 assembler 的連續順序提供此不變量，不能允許其他 caller 跳號發布。
- 子程序早退造成的 pipe error，來源主要在 finish 才回報；失敗期間仍可能繼續解碼。
- abort 先 kill child，再喚醒／join writers，可讓 blocked pipe write 結束；正常 finish 則可能等待不再進展的 child，不能宣稱已有完整 cancelable finish。
- `SIGPIPE` disposition 是 process-global；來源在整段 streaming 期間忽略、結束後還原。較長存活期間及多個 concurrent streams 的交錯，必須符合本機宿主的 signal ownership。
- partial filename 僅由 PID＋basename 組成，同程序同目的地並行工作會碰撞；若宿主允許這種呼叫，不能直接採用。
- 來源已有 synthetic assembly 與可選 FFmpeg byte-comparison tests，但前者共用抽取後 helper、後者有環境條件；**測試存在不等於已在本機通過**。

### 3.4 audio／duration／lipsync 不能靠「早送影格」自行對齊

來源 `spawn_av_encoder()` 保留本機命令的核心參數：

- video：RGB24、固定 `-framerate`；
- audio：F32LE、原 sample rate／channels；
- libx264 `fast/crf18`、AAC `192k`、`+faststart`；
- **沒有** `-shortest`、`-t`、補靜音或 resample。[VF][BF]

本機 AudioVAE 輸出為 F32 stereo，sample rate 32000，`samples=audio->length`。streaming 只做相同 planar→interleaved copy，不應改 sample count。[BA]

因此必須分別保留：

- 影格索引與 `F/fps`；
- PCM bytes、sample count 與 `samples/sample_rate`；
- 原有 A/V 長度差，而非偷偷 trim／pad；
- AAC priming／尾端 padding、packet PTS/DTS、最後 mux flush 與 `+faststart` 完成。

Rawvideo arrival 的 wall time 不是應新增的 timestamp。作者聲稱「相同 frame／PCM／command，因此檔案 bytes 相同」尚需綁定實際 FFmpeg build、encoder 行為及完整 mux tail 驗證；同輸入是必要條件，不是跨環境的檔案等值保證。

**影片暫停時，能驗證的是合成資料的交付／計數／所有權；不能認證中文聲音、台詞、嘴型或 AV 感知品質。**

---

## 4. 記憶體帳：雙槽不是整套 scratch ×2，也不是零額外 residency

以下是 **[INFERENCE：依已讀配置式推導]**，不是 peak 量測。

令 tile latent patches 為 \(P=7h_tw_t\)，sequence \(S=P+5\)，整幅每幀 RGB 元素數 \(C=3WH\)。

來源額外配置：

- latent slots：`2 × 4 × 24P`；
- 額外 projected slot：`4 × 3072S`；
- host input staging：`4 × 24P`；
- zero suffix：`4 × 2048`；
- assembler：
  \[
  4\left(22K\,T_hT_w\,3+27C\right)
  \]
  其中 K 是 spatial tiles 數，27 是 22 幀 stitched chunk＋5 幀 temporal overlap；
- 全段 RGB24：`FC`；
- PCM 原 buffer＋interleaved copy；
- 另加 FFmpeg child、兩條 pipes、encoder／mux buffers、MPSGraph workspace 與仍活著的 DiT。

### 768×1344、141 幀的具體例子

**條件：**沒有 tile override，沿用本機 `configured_tile_pixels()` 的面積評分；由程式推得 320-pixel tile、3×5=15 tiles，P=2800、S=2805。這不是重新選 tile 的建議。

| 項目 | 推導配置量 |
|---|---:|
| VAE 36 層四個大矩陣的 BF16 weights | **4.5 GiB**，另加小型 F32 weights／bias／norm |
| 本機既有主要 VAE activation scratch | 約 **0.675 GiB**，不含 framework workspace |
| projected 兩槽合計 | 約 **65.74 MiB**；其中一槽原已存在 |
| 新增兩個 latent slots＋額外 projected＋host staging＋zero | 約 **33.65 MiB**；舊 latent 如仍保留須另計 |
| 全 chunk tile RGB | 約 **386.72 MiB** |
| stitched 22 幀＋temporal overlap 5 幀 | 約 **318.94 MiB** |
| 141 幀完整 RGB24 | 約 **416.39 MiB** |
| 可由 stream API 避免的完整 F32 RGB | 約 **1.627 GiB 配置量**，不是已量到的 physical-peak 降幅 |

來源 fused path **仍配置許多已不再走熱路徑的 F32 intermediates**；不能把少寫它們等同於已移除 residency。來源 allocation 還保留 F32 `heads`，而本機 BF16 分支不配置它；不得整段覆蓋本機 allocator。[VS][BV]

上述可見 VAE／assembly／RGB24 項目約在 6.3 GiB 量級，**尚未計齊**模型載入重疊、DiT、audio、framework／driver／allocator 保留及 FFmpeg child；不是總量上界。

**40 GiB soft budget 的決策：**

- 不把兩套完整 VAE scratch／weights 複製成雙槽。
- 不同時引入 prefetch、early-load、queue-depth 放大。
- 首版不採來源的 asynchronous single-use DiT free；它可能讓尚未釋放的 DiT 與 VAE／audio 重疊。
- cached DiT 原本就可能保留；不能為省帳偷偷破壞 resident reuse。
- 同時看 parent＋FFmpeg child footprint、Metal live bytes、workspace 與全機壓力，保留 LLM／OS 空間。隔離的 6.776 GiB 結果不能替代這項驗收。

---

## 5. `4aed086c` 與 dense 反證的新增判讀

### 5.1 診斷 patch：有用，但 counter 名稱比實際證據強

`h3_gpu_adopt_mps_root()` 在 SDPA、conv、linear、MLP、GQA 的 MPS encode 後：

```text
root != gpu.command → mps_commits++
mps_encode_seconds += elapsed
gpu.command = root
```

本機本來就有最後一步 root adoption；新價值是前兩項及 `h3_gpu_continue()` 的 command-buffer creation 計時。[D][BG]

但：

- `mps_commits` 是**一次 encode 前後 root 改變的觀察次數**，不是 encode 內所有 commit／child buffers 的精確總數。
- `mps_encode_seconds` 包含 feed／result preparation 與 encode wall time；`command_create_seconds` 包含 creation 成本與可能的 queue-limit wait。兩者不是純 GPU 執行時間。
- encode／queue-create 是外層 encode 時間的部分，不可重複相加。
- wait time 也不等於完整工作量時間：先前 GPU 已與 host encode 重疊執行。
- 來源 `submissions + mps_commits` 不應標成已精確捕捉「所有 command buffers」。

建議名稱使用 **observed root replacements／MPS encode wall／command-create wall**，保留 root-only GPU timing 的限制；完整 phase wall 從開始到所有依賴退休獨立量測。

`H3_GPU_QUEUE_DEPTH` 透過 `newCommandQueueWithMaxCommandBufferCount:` 改所有 contexts。它不改算式，但會改 host 阻塞位置、in-flight resource residency，以及跨 contexts 的提交交錯；**不能當成沒有副作用的純診斷欄位**。不提議改預設，也不把作者「same order」擴張成跨 queue／CPU／LLM 的整體排程不變。

### 5.2 danmo：具體可借鏡，沒有可採用的 dense 勝出

已讀 `h3_gpu_flash_attn_nax_impl()`、`h3_flash_attn_nax_bf16_impl<32,4>`、`h3_nax_mma()`：

- 每 threadgroup 4 simdgroups，各 16 query rows，共 64 rows；
- D=128、BK=32；
- BF16 Q/K/V、F32 scores／running max／sum／output accumulator；
- PV 使用 `h3_nax_mma<float,bfloat>`，不是先把 probabilities 降 BF16；
- 用 `fast::exp2`、不同 reduction／matmul accumulation，不能稱與 MPSGraph bitwise；
- dense mode 遍歷全部 keys；CSR sparse mode 是另一條 connectivity 契約。[DN]

作者 S=29584 dense `642–672 ms` 對 MPSGraph `555–580 ms` 是負面證據。其測試實際比較 scalar flash，而非檔頭註解所稱的 MPSGraph oracle；不能替代本案 production-N baseline／F64 驗證。Q resident 導致 spill 的作者觀察可作排程研究參考，不是目前機器的測量。

### 5.3 qpzmV：名稱有 NAX，固定 kernel 卻不是 TensorOps matmul

`h3_sdpa_p1` 是一 lane 一 query row，維持 `float cO[128]`、`float S[64]`，以 scalar 迴圈計算 QK／PV、log2-domain online softmax。`h3_gpu_sdpa_nax_try()` 僅接 row-major，明確拒絕目前主要 head-major producer／consumer 路線。[QPS][QP]

作者 23–46×較慢的範圍及 opt-in 修正支持不採用。可以借鏡完整 K/V traversal／tail handling；不能由後續 cooperative-stage commit metadata 推導最終效能勝出。

### 5.4 neokree：與 PR77 的真正差別及 oracle 限制

`23fa7273` 不是單 graph 的 slice＋concat：

- `h3_gpu_sdpa_graph()` 區分 `queryRows`／完整 KV shape；
- host 逐 block encode；
- `h3_gpu_graph_data(..., offset)` 使用 MPSNDArray offset view，非零 offset 不走 stable cache；
- 僅 batch=1、noncausal、row-major input/output、row bytes 256-aligned 才 blocking；head-major 保持 whole-sequence；
- 預設 block=256，且沒有避開小 VAE attention 的 threshold。[NQ]

所以它是**不同實作參考**，卻不適合為本機 head-major 路徑直接搬入；也沒有足夠證據值得重開已封存的速度實驗。

`oracle_row()` 對抽查 query 計入所有 K/V；production case 是 **64 rows × 8 heads × 全部 128 channels**，另做 block variants 全 tensor byte comparison。`d788ad2d` 把錯用 CPU `clock()` 的計時改成 monotonic wall，值得保留這項教訓。[NO][NM]

不接受來源的兩個過度推論：

- 完整 F64 oracle 不必配置完整 \(N^2\) scores；可以逐 row 算。其問題是計算成本，不是「497 GB 因而不可能」。
- offset off-by-one 不是唯一可能 bug；還有 dtype、head stride、tail、cache view 與同步。
  
作者約 `2.228e-3` rel-L2 與 block 間 bitidentity，不是本機 baseline 等值／感知品質門檻。

---

## 6. 排序後的四個新增量 patch

**以下是主程序另行核准後的非影片驗收提案，本次全部未執行。排序按可證明性與改動風險，不按未量測加速。**

### ① 診斷限定：root replacement／encode／queue-create 歸因

- **範圍：**`h3_gpu.m` 現有 MPS encode seams、`h3_gpu_stats`。
- **不取：**queue-depth knob／預設更動、額外 submit／wait、完整 child-time 宣稱。
- **正確性 gate：**合成 linear→SDPA→linear chain，診斷開關前後 bytes 相同；direct-only 與 MPS paths 均覆蓋；統計不改提交次序。
- **效能 gate：**量診斷自身 overhead；phase wall、root GPU、encode／create／wait 分列，不能加成假總量。
- **記憶體：**只加固定大小 counters，不保留無界 command history。
- **授權：**保留 Bambushu 固定來源與 MIT／既有適用 notices。

### ② VAE 等算術 epilogues；先保持原逐 tile blocking

- **範圍：**本機 `project()`／`run_block()`、相應 Metal kernels。
- **順序：**RMSNorm→BF16；bias＋SwiGLU；bias＋residual；最後才考慮重複 bias 成本較高的 QKV fusion。各項獨立 A/B，不與排程綁包。
- **正確性 gate：**合成 tensors 覆蓋 width 2048／8192、32×64 QKV、RoPE 24／48 邊界、實際 tile rows 與 ragged rows、BF16 midpoint、抵消值／極值；逐 stage BF16 bytes、F32 residual bytes 必須相同。保留原 norm reduction threads、matmul／SDPA primitive 與 safe-math 設定。
- **效能 gate：**既有 primitive 組合與融合版的完成後 wall，cold／warm 分開、平衡交錯。QKV 若增加算術抵銷 traffic 收益，就拒絕該子項。
- **記憶體：**初期不把「減少讀寫」宣稱為 allocation 降幅；沒有必要複製來源整包 INT8-VAE／cache allocator。
- **界線：**不觸及 DiT beta5 final／audio。

### ③ VAE 雙槽退休與 host assembly；先不接 FFmpeg

- **範圍：**`run_resident_tile()` 拆 encode／wait；兩個 input/output slots；重用 suffix；原 unpack／stitch／temporal blend。
- **介面：**新增最小、唯讀 shared-buffer view 僅供**已退休 output slot**消費；其 lifetime 必須清楚，不搬入來源全部 cache API。
- **正確性 gate：**合成 projected rows 對**未改寫 baseline assembly**比較；單 tile、多 tile、非均勻 overlap、單／多 chunk、17＋5 尾段；兩槽交替、consumer 停止、command 建立／encode／completion 失敗後均無提前重用或 free。兩-token診斷仍走原路徑。
- **效能 gate：**無模型的 synthetic producer＋真實 unpack／stitch，分開觀察 host overlap 與額外 memory cost；這只能證明排程機制，不能當真實 VAE 加速。
- **記憶體：**僅 I/O slots 加倍，共用 scratch／weights；queue 預設不變；不非同步 free DiT。

### ④ 最終影格 sink／RGB24 發布，再評估 FFmpeg feeder

- **範圍：**先從同步 tile 路徑也可交付 final chunks，與 patch③解耦；生成、cached decoder、既有 latent-decode caller 的行為分別保留，不因新增 stream API 改掉其他輸出契約。
- **非影片 gate：**合成 F32→RGB24 bytes、連續 prefix、慢 pipe consumer、短寫／EINTR／EPIPE、取消與 child-failure ownership；涵蓋 no-output-path、callback 取消與最後五幀。PCM interleave bytes／sample count 必須完全相同。
- **效能 gate：**只量 conversion／pipe 背壓／交付 wall；若保留全 RGB24，就明列其 O(F) residency，不稱 bounded ring。
- **尚未可放行：**真實 FFmpeg container bytes、packet timestamps、AAC 尾段及 AV 品質。非影片 harness 不能替代實際 mux；影片暫停期間保持未認證。

---

## 7. 授權、read ledger 與剩餘缺口

### 授權處置

- 已讀 Bambushu、danmo、qpzmV 固定版本的 `LICENSE`：均為 MIT、Salvatore Sanfilippo copyright；其 `THIRD_PARTY_NOTICES.md` 均列 ccv／Liu Liu **BSD-3-Clause**。
- **danmo 的新 kernel 明確註記跟隨 MLX `attention_nax`、MIT／Apple Inc.，但該固定版第三方 notices 只有 ccv。**不能只帶根 MIT 就宣稱完成來源義務；若複製，需補核其實際 MLX／Steel 衍生檔案與適用 Apple copyright／permission notice。本次沒有完成這條 upstream genealogy。
- qpzmV 此固定 per-lane kernel 未呈現同一 MLX register-fragment 實作；不能把 danmo 的 MLX 歸屬泛套所有「NAX」來源。
- Bambushu VAE epilogues 的已讀區段未見 MLX／Steel 新來源標示；保留 h3.c MIT、PR73 provenance 及既有適用 ccv notice，不替未讀依賴作全面授權保證。
- 程式 MIT 不授權 Eros／MiniMax／LightVAE 權重或 derived caches。模型與 adapter 條款維持獨立；不因本次 host/kernel patch 改變。[BL][DL][QL][BN][DN-N][QN][B-N]

### 實際閱讀紀錄

| 來源 | 本次讀取範圍 |
|---|---|
| Astra #1 保存文件 | **全文 1–709 行**，包含主程序後續結果 addendum |
| Luna forks/platforms 文件 | 全文；authenticated supplement 126 行起另以 raw 完整讀取，補足長行 |
| quality-safe contract | `local://` 不可用；已找到並完整讀取原 session `local/h3-quality-safe-contract.md` |
| discovery artifacts | coverage 欄位、compact index 結構／baseline record；**沒有重新遍歷或重算全部 discovery** |
| `b0966252` | **完整 2406 行 diff**；另讀固定 VAE allocation／load／unpack、GPU continue／submit／wait、compiler options、buffer-view API、FFmpeg argv、license／notices |
| `4aed086c` | 完整 diff：所有 root-adoption callsites、stats、queue creation、VAE profile |
| `02dd9c72` | 完整 diff：kernel／host dispatch、author perf note、benchmark、correctness test；固定 license／notices |
| `a13b938c` | 完整 diff；固定 `h3_gpu_sdpa_nax_try()`／caller 與 `h3_sdpa_p1` kernel；license／notices |
| neokree 三項、Bambushu opt-in revert | 全部指定 diffs，含 oracle 與 timing 修正 |
| 安裝版 SSH baseline | VAE `project/run_block`、allocation、tile plan、decode routing／assembly；GPU command／rounding／compiler／buffer APIs；`h3_generate` tail、FFmpeg writer、AudioVAE output、notices |

**仍未證明：**新 epilogues 的本機 byte parity、MPS child completion/error 全涵蓋、雙槽 fault/cancel drain、實際尾段 peak／收益、FFmpeg mux 等值，以及任何感知品質。

OS 27.0.1／26A434、SDK 27.0、CLT 與 runtime Metal compilation 的主程序證據仍有效；**specific cooperative-MSL feature gate 尚未完成，也沒有 full Xcode 認證**。這不妨礙保守評估普通 epilogue fusion，但不能替新 cooperative attention kernel 背書。

**最終處置：保留現有正式 baseline；新增候選只按上述四項拆分驗收。沒有任何新來源取得本機加速或「身份／動作／中文聲音／台詞／嘴型／AV 無可察覺退化」認證。**

### 引用

[A1]: eros-beta5-astra-luna-portability-2026-10-26.md
[L2]: eros-beta5-luna-forks-platforms-2026-10-26.md#authenticated-full-branch-supplement--luna-consultation-2
[V]: https://github.com/Bambushu/h3.c/commit/b0966252a375fb464f4046276f63af53b8ccc626
[VS]: https://github.com/Bambushu/h3.c/blob/b0966252a375fb464f4046276f63af53b8ccc626/h3_video_vae.c
[VF]: https://github.com/Bambushu/h3.c/blob/b0966252a375fb464f4046276f63af53b8ccc626/h3_ffmpeg.c
[D]: https://github.com/Bambushu/h3.c/commit/4aed086cd8e116783fad7d32936e8b872599778a
[DN]: https://github.com/danmo-ai/h3.c.ui/commit/02dd9c72612340832acdbac04c0235f1edf71653
[QP]: https://github.com/qpzmV/h3.c/commit/a13b938c4deb3d16080bd6fbfa0d8139a7fc15bf
[QPS]: https://github.com/qpzmV/h3.c/blob/a13b938c4deb3d16080bd6fbfa0d8139a7fc15bf/h3_shaders.metal
[NQ]: https://github.com/neokree/h3.c/commit/23fa7273b3ed0ee5d2e3126b8afcd012b27702ea
[NO]: https://github.com/neokree/h3.c/commit/90624ca485a97af8871facdd91e4f3d80af96e8a
[NM]: https://github.com/neokree/h3.c/commit/d788ad2d36ad72dccc42aab97b9055ec8c3be397
[RV]: https://github.com/Bambushu/h3.c/commit/97bca6e9ecfbe6e55b61ac33d4956e932b3375fb
[BV]: ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_video_vae.c
[BG]: ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_gpu.m
[BS]: ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_shaders.metal
[BC]: ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3.c
[BF]: ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_ffmpeg.c
[BA]: ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_audio_vae.c
[B-N]: ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/THIRD_PARTY_NOTICES.md
[BL]: https://github.com/Bambushu/h3.c/blob/b0966252a375fb464f4046276f63af53b8ccc626/LICENSE
[BN]: https://github.com/Bambushu/h3.c/blob/b0966252a375fb464f4046276f63af53b8ccc626/THIRD_PARTY_NOTICES.md
[DL]: https://github.com/danmo-ai/h3.c.ui/blob/02dd9c72612340832acdbac04c0235f1edf71653/LICENSE
[DN-N]: https://github.com/danmo-ai/h3.c.ui/blob/02dd9c72612340832acdbac04c0235f1edf71653/THIRD_PARTY_NOTICES.md
[QL]: https://github.com/qpzmV/h3.c/blob/a13b938c4deb3d16080bd6fbfa0d8139a7fc15bf/LICENSE
[QN]: https://github.com/qpzmV/h3.c/blob/a13b938c4deb3d16080bd6fbfa0d8139a7fc15bf/THIRD_PARTY_NOTICES.md


### 主程序後續澄清：K1024 不等於 weight prepack

主程序在第二次諮詢完成後明確澄清：既有 FC2 K1024 實驗 **沒有實作 weight prepack**。被測且拒絕的是 K1024 kernel variant；真正 weight prepacking 尚未實作、未量測。不得從 K1024 結果推導 prepack 已測且無收益，也不得宣稱 prepack 已有收益。Luna 補充表第155行已由主程序更正；本文件保留實際 Astra 回覆原文，以此補記消除可能的混淆。

