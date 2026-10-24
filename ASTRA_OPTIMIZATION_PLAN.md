# Astra 建議與實作裁定紀錄

> 前段保留歷史提案，部分方案經量測調整或未採用，不是當前待辦清單；末段為2026-10-25工具核查後的建議。使用者已決定暫時收案，等實際流程回饋再重開，不主動執行後續候選。日常用法見[README](README.md)，實作與量測見[COMFY_CACHE_RESEARCH](COMFY_CACHE_RESEARCH.md)。下文「本輪／當前」依各次提案當時的快照解讀。

**建議本輪實作兩項主改動、一項量測後決定的改動：**

1. **LoRA 合併時，避免每個 output-row tile 重複解碼 A 因子。** 直接改善第一次生成，不依賴暖快取。
2. **將 resident DiT 權重與每次生成的 conditioning／schedule／workspace 分開失效。** 改 prompt、steps 或尺寸，不應一律重新合併及量化全部 core weights。
3. **量測每個 DiT block 合併量化提交的收益。** 若確有收益，將目前三次同步提交縮成一次；不改量化演算法。

VAE 是重要的第二戰場，但**本輪不能把微小配置清理或尚未驗證的 BF16／INT8／Lite 替換，包裝成已解決40秒解碼成本**。

本報告只讀來源與程式碼，沒有修改檔案、建置、測試或推論；以下收益均為待主代理驗證的假設。

---

## 一、證據與一項重要更正

來源對話的實際768p、39幀、Lightx v1.2五步紀錄：

| 階段 | 實際秒數 |
|---|---:|
| Qwen | 4.402 |
| DiT load／prepare | 51.035 |
| DiT denoise | 50.722 |
| Video VAE | 39.991 |
| Audio VAE | 0.387 |
| 整體 | 148.968 |

這只能證明三個主要成本區塊，**不能證明51秒主要花在SSD或LoRA合併**。來源對話已明確指出細項尚未隔離。

### VAE 現行運算並不是 BF16

研究筆記將 native VAE 簡稱為「原始BF16」容易誤導實作。當前程式實際是：

- `h3_video_vae.c:load_f32/f1/f2` 呼叫 `h3_weight_load_f32`。
- `run_block` 使用 `h3_gpu_linear_f32`、`h3_gpu_sdpa_f32`、F32 norm、SwiGLU及residual。
- `h3_gpu.m:h3_gpu_linear_f32` 對大型矩陣已有 **MPS F32 linear** 路徑，並非全部使用慢速 scalar shader。
- VAE目前36層，tiled path 已在該解碼階段保留所有權重，不會每個tile重讀全部權重。

因此，**改成BF16 VAE也是數值策略變更，不是單純把現有BF16運算接到較快kernel**。要另做畫質與數值驗證。

---

## 二、第一優先：消除 LoRA A 因子的重複轉換

### 已確認的重複工作

`h3_lora.c`：

- `ROWS=128`、`RANK=64`。
- `h3_lora_load_target` 每128個output rows呼叫一次 `h3_lora_merge_rows`。
- 後者對每個factor、每64-rank panel，再將相同A的BF16/F16資料轉成F32。
- 只有已對齊的F32 A能直接使用來源指標。

因此，相同 `A[rank, columns]` 的轉換會隨output-row tiles重複。這符合主代理來訊指出的hot path。

### 最小實作建議

在**載入目前這一個target期間**：

1. 為需要轉換的A建立對齊F32暫存。
2. 後續所有128-row tiles共用該A。
3. 保持：
   - factor canonical順序；
   - 每128 rows、每64 ranks的SGEMM呼叫順序及形狀；
   - scale、beta與delta累加方式；
   - base最後相加；
   - BF16最後只round一次；
   - B的QKV row mapping。
4. target載入完成即釋放，不把全模型LoRA升成F32。

**不要交換成「先掃rank panel，再更新整個base矩陣」**，那容易改變累加與rounding語義，也可能需要巨大F32 delta。

### 記憶體契約

暫存量應以目前target的A總量計算，而不是模型全部factor總量。多LoRA仍須有明確上限，不能因任意adapter數量變成無界記憶體成長。

可採一個內部固定預算：

- 預算內的已轉換A可被重用；
- 超過預算的factor使用原有有界panel解碼方式；
- 不改成功／失敗及數學語義，不增加使用者參數；
- 分配失敗沿用清楚的錯誤處理，不把錯誤掩藏為「已最佳化」。

既有公開 `h3_lora_merge_rows` 的直接caller仍須正確；production loader可以共用一個private merge核心，**不要複製兩套合成演算法**。

### 預期與證明界線

[INFERENCE] 這會消除已確認的重複轉換成本，值得第一個實作；但SGEMM、讀取或其他precompute若佔主要比例，端到端收益可能有限。**不可先承諾51秒減半。**

優先驗證：

- 同一target所有merged bytes一致；
- F32/BF16/F16、未對齊snapshot、rank尾端、QKV排列、多LoRA及signed strengths；
- 舊／新完整生成的RGB與PCM一致；
- 附加暫存峰值與實際cold wall改善。

---

## 三、第二個冷啟動候選：減少量化同步邊界

### 程式證據

`h3_dit.c:load_core` 每層依序呼叫：

- `quantize_block_mlp`
- `quantize_block_qkv`
- `quantize_block_attention_out`

每個函式各有 `h3_gpu_begin`／`h3_gpu_submit`。`h3_gpu.m:h3_gpu_submit` 會等待inflight command buffers完成。

對同時啟用這三項量化的50層路徑，這是**150個量化提交／等待點**，不含其他準備階段。

### 建議

使用同一層的量化準備函式：

1. 分配該層所需int8與scale buffers。
2. 同一command chain編碼既有四個weight量化操作：FC1、FC2、QKV、out。
3. 一次submit並等待完成。
4. 完成後才釋放不需保留的BF16 buffers。
5. 原來的診斷保留旗標及非M5／BF16選項照常生效。

不改quantizer、不改scale、不跨層無限制排隊；SSD路徑不受影響。

[INFERENCE] 此改動可能降低同步開銷，但量化本身若主要是記憶體帶寬，收益也可能很小。**須單獨量測，不能與A預解碼同時落地後才猜各自貢獻。** 另檢查一次分配更多int8輸出是否增加該層峰值。

---

## 四、模型常駐：保留單一 DiT，重新準備請求，而不是建立新服務

### 已確認問題

`h3.c:h3_prepared_key` 將conditioning字串、shape、steps及其他選項綁一起。key改變時 `h3_generate` 直接 `h3_dit_free(ctx->dit)`。

但 `h3_dit_free` 同時釋放：

- 50層已合併／量化的core weights；
- GPU instance與其linear／MLP／attention等cache；
- prompt refined text；
- schedule、maps、RoPE與activations。

Comfy可借鏡的是**不同依賴的狀態分開失效**，不是照搬Python graph cache。

### 最小設計

保留既有 `h3_dit` owner與GPU instance，在其內部分出：

- **resident部分**：core weights及與模型身分／實際精度相關的不變資源。
- **prepared request部分**：layout、refined text、schedule、maps、RoPE、activations、condition rows及run state。

新增一個內部重新準備入口，例如 `h3_dit_prepare_request(...)`；名稱由主代理沿用現有慣例決定。冷載入與暖重新準備必須共用準備邏輯，避免兩套流程。

**不要只從key移除prompt。** 現有 `h3_dit_reset_run` 只更新condition rows及core-reuse計數，沒有重新refine text，也沒有重建layout、schedule與buffer尺寸，不能拿來處理prompt或shape變化。

### 建議失效表

| 修改 | 可保留 | 必須更新 |
|---|---|---|
| 只換seed | 權重、conditioning、prepared request | 新noise、condition augmentation、run history |
| 換prompt | 不變core weights、GPU instance | Qwen、refined text、text相關layout/maps及必要workspace |
| 換steps／shifts | core weights、未變的conditioning | sigma、AdaLN schedule、step相關maps、run history |
| 換shape／frames | 同身分且實際精度相容的core weights | references相關conditioning、layout、RoPE、workspace及maps |
| 換參考素材 | core weights | 相關conditioning、layout及condition rows |
| 換base／LoRA stack／精度配置 | 不保留不相容權重 | 完整重建 |

### 不能漏掉的細節

- **QKV／attention-out的int8啟用條件含 `sequence >=128`**。shape或prompt token數改變可能跨門檻；resident相容性須使用實際選定精度，不能只比CLI旗標。
- `configure_gate_ranked_blocks` 使用schedule gate分數選層。減層路徑換schedule後可能需要不同block；不能重用錯誤的active set。最簡單是相容時保留，實際active set改變則走明確重建。
- AdaLN目前逐block讀取大型projection並釋放；`h3_dit_schedule.h` 明文每個projection約498MiB。**不要為省重建schedule就保留全部projection，這可能額外增加數十GiB。**
- 相同sigma及condition-modality狀態下可沿用schedule；steps／shifts或condition種類改變則重算，這仍然可以避免重載大部分core weights。
- activation alias指標必須先解除再釋放；`h3_dit_free` 已有對應例子。
- step maps必須按**舊schedule的step數**清理，再替換schedule。
- 重新準備失敗或取消後，不得把半完成狀態標成cache hit；可釋放整個DiT並返回失敗，下一次正常重建，不能使用舊prompt繼續生成。
- `h3_cache_clear`／disable cache仍清除resident與prepared資源，LoRA immutable snapshots依原契約保留。
- 不新增全域跨context權重共享或LoRA hot-swap。

### 簡單使用方式

原有互動入口 `./h3 -d ...` 已在 `h3_cli.c` 開啟cache。延用它：

- 第一支完成後保留模型。
- 改prompt／seed／steps由程式自行處理失效。
- `!cache`／既有status即可說明是否保留，不加cache層級選單。

**一次性 `-p` 程序結束後不能繼續持有模型。** 沒有daemon就不可能跨獨立程序永久常駐；不應對使用者宣稱兩者同時成立。本輪不需要新增daemon或服務管理。

---

## 五、VAE：先承認現有優化，再選真正有意義的路線

### 現有路徑已經做了什麼

`h3_video_vae.c`：

- `load_resident_weights` 已一次載入所有decoder weights。
- `run_resident_tile` 一個tile編碼完整36層再submit。
- `configured_tile_pixels` 已在256–320之間選擇tile。
- `decoder_decode_chunk` 對每個tile完成輸入準備、解碼與CPU讀回，再stitch。

所以不建議將「保留VAE權重」當作新發現；此能力已存在。每tile的2048-F32 zero allocation可順手消除，但不是40秒的核心方案。

### 三條路線的定位

**A. 不改數值的配置／資料搬移改善**

可重用latent staging及zero buffer，減少每tile `extract_latent_tile → prepare_input → GPU tensor` 的配置與複製；維持同一tile、同一運算順序及stitch結果。

[INFERENCE] 可減少CPU配置開銷，但既有紀錄VAE等待時間很大，不能期待僅此產生主要提升。

**B. 固定tile語義的多tile batching**

可能改善矩陣利用率，但目前 `run_block`／`h3_gpu_sdpa_f32` 沒有直接表達多個獨立tile的batch隔離。不能把tiles直接接成更長sequence，否則會跨tile注意力。

需要真正batch-aware attention、RoPE、register/suffix與輸出拆分；也可能改變GEMM累加及峰值記憶體。**不屬於這輪最小安全scope。**

**C. LightVAE／BF16／INT8：獨立品質取捨項目**

LightVAE Lite最值得作為下一個明確加速候選：少做decoder blocks而非只調小配置。但需要正確checkpoint metadata、retained-block mapping、normalize與temporal stitching契約。

- 優先只替換decoder，保留原參考encoder及audio VAE。
- 不得以原始權重任意跳層冒充已訓練Lite。
- 不默認開啟、不以CUDA倍率預告M5速度。
- 若要接入，最多一個清楚的decoder選擇入口，不暴露一串後端旗標。
- 同latent比較原decoder與Lite，避免重新去噪造成品質差異難以歸因。

同樣地，native F32 VAE改BF16或int8必須當成數值變更；現有DiT的int8 API不代表VAE的bias、norm、shape與精度要求已經適配。

---

## 六、主代理的驗收矩陣

### 1. 冷啟動成效不能用暖快取冒充

固定同一模型、LoRA hash、prompt、seed、尺寸、幀數、sampler及steps：

- 舊／新各用新程序。
- 交錯順序並重複，報告median與變動，而非只挑最快一次。
- 分別記錄LoRA讀取／驗證、A轉換、merge、量化、AdaLN、去噪及VAE。
- 同時報整體wall、physical footprint、swap觀察。
- 說明新程序不等於OS page cache也被清空，不需破壞性清理系統cache。

預設合入門檻建議：**代表性冷啟動案例改善大於量測變動，且非LoRA路徑無明顯退步。** 若收益低於噪音，不能宣稱達成「單次加速」。

### 2. 數值驗收

對預期無損的A預解碼、量化提交整合及cache重準備：

- merged tensor bytes；
- int8 weights及scales；
- 相同輸入的denoised latents；
- 解碼後RGB與PCM。

以bytes相同為目標；MP4容器hash不適合作為唯一數值證據。若bytes不同，先找原因，不能臨時改成寬鬆SSIM門檻就宣布無損。

### 3. 常駐序列

在同一context依序測：

1. 初次生成。
2. 同prompt換seed。
3. 換成token長度不同的prompt。
4. 改steps，再改回。
5. 改shape／frames，再改回。
6. 增減／替換Ref2VA圖片、影片或音訊。
7. cache clear後重建。
8. 取消後再次生成。
9. 故意準備失敗後恢復有效輸入。

每項和**相同新設定的fresh context結果**比較。不能只檢查「沒有load log」；須確認結果不是舊conditioning、舊maps或舊sampler history。

涵蓋FL2VA、Ref2VA多模態、無LoRA／零strength／多LoRA、simple RES／LCM及既有serving，並保留target錯配、SSD、anchor互斥等負向驗證。

---

## 七、本輪範圍裁定

**立即實作：**

- 有界、per-current-target的LoRA A預解碼重用。
- 同一DiT owner內的resident／request失效分離。
- 同步補上針對新行為的測試與簡單CLI文件。

**量測有實益才合入：**

- 每block量化提交整合。
- VAE staging小型配置改善；不得當主要成果。

**不混入本輪預設：**

- LightVAE、F32→BF16／INT8 VAE。
- 改tile尺寸、任意減層、降解析度、減steps。
- CUDA ConvRot／Sage移植。
- 持久化merged checkpoint、跨程序disk cache、全域model registry、daemon、LoRA hot-swap。

這樣能用最小可維護改動，同時回答使用者兩個不同需求：**第一次真的少做重複準備工作；後續改prompt時，不再丟掉已準備好的大模型。** VAE的大幅提升另有清楚路線，但在實測前不承諾速度或畫質。

## 執行裁定與額外正確性發現

以上為 Astra 的初始方案；以下是依實測及後續 Astra 諮詢作出的實作裁定：

- **撤回 A 預解碼快取**：第一組完整 cold wall 沒有可辨識收益，因此不保留。四個真實 large target 的全部 merged bytes 曾與原 panel 路徑一致；未證明該 cache 是後來影音差異的根因。
- **改採兩個有界 row workers**：在單一 large target 內以連續128-row區間並行，仍使用原64-rank SGEMM順序；私有scratch/tile/error、唯讀factors、`pread`、不重疊range寫入，join後才量化。兩份workspace須同時落在64MiB預算內，small target保持serial。Astra後續審視確認此路徑可作實測候選，不改全程序BLAS thread設定。
- **同block量化只submit一次**：保留FC1、FC2、QKV、out原kernel與順序、精度與keep flags；GPU完成才釋放來源。冷啟動報告僅對兩項合併效果作結論，不虛構個別貢獻。
- **resident／request失效分離**：原有互動入口自動保留相容的50層DiT與GPU owner；request重建共用冷載入的準備邏輯。相同video/audio sigma clocks與conditioning-modality presence可沿用AdaLN schedule。SSD／token reduction／減層及小sequence精度轉換保持完整重載。
- **修正互動caller的過度清除**：修改reference／anchor不再無條件`h3_cache_clear`；由owned輸入簽章與resident mode key處理。明確`!cache clear`仍完整釋放，沒有新cache旗標、daemon或全域registry。
- **數值回歸揭露既有scratch競態**：Ref2VA fresh／clear／shape測試中，原始video/audio conditioning一致，但Qwen text embedding不同。獨立vision feature重複六次一致，進一步定位到causal GQA的max→sum reduction缺barrier。fixture-free GPU oracle在168 tokens觀察max-error **0.746718**，不是可忽略的BF16誤差；修正後softmax及GroupNorm/SiLU CPU oracle與重複性檢查通過。相同共享scratch問題也存在visual encoder GroupNorm的mean→variance階段，兩處均補上必要barrier。
- 修正後同一conditioning probe的fresh／clear／shape／back五次text/video/audio bytes一致。這是正確性修正，不是為了加速降低精度；不同於錯誤kernel的舊結果不應被當成品質退步或靠寬鬆SSIM掩蓋。

正式冷／暖數字、完整fresh-versus-warm矩陣與交付證據集中在 [COMFY_CACHE_RESEARCH.md](COMFY_CACHE_RESEARCH.md)，避免把提案中的假設寫成已量測收益。LightVAE／BF16或INT8 VAE、CUDA稀疏注意力與持久化disk cache均未混入本次預設。

## 2026-10-25：綜合應用與下一階段建議（Astra唯讀核查）

使用者要求：「並且效能優秀? 還有明顯的優化空間嗎? 專案整理下 你提的幾個已經算是進階應用?」「就是指綜合應用 實際上用指令分開執行也是可行的吧?」，並指定交給Astra提出建議。第一輪先將原話、實作狀態、精確效能與模型清理界線交給**實際`llm-esapp/gpt-6-astra`**；收到回覆後，依使用者補充再開放read／grep／glob／web_search做第二輪唯讀核查。第二輪完成65次唯讀工具呼叫，沒有改runtime、下載權重、build、tests、SSH或新增推論。下列是諮詢建議，**不是已完成的新功能／benchmark，也不自動授權實作**。

完整brief與兩輪原文在本機`/Users/mics/Movies/H3-Astra-Review-20261025/`：`request.md`、`01-initial-advice.md`、`02-investigation-request.md`、`03-investigated-advice.md`；`consultation-verification.json`記錄實際model與唯讀呼叫。第二輪CLI外層回傳1／yielded，但確實收到完整final assistant文字且model=`gpt-6-astra`、stopReason=`stop`；不將該CLI退出碼當作模型測試通過。

### 結論與對原排序的修正

**常見綜合用途已可用CLI分步完成；不必重建ComfyUI。** 批次挑片、多參考、分鏡剪接、外部字幕混音放大是編排，不應當成引擎缺少進階功能。獨立process仍重付cold成本；同模型／adapter反覆試片應使用現有互動context。mask／ControlNet／任意時間guide／latent import-export／partial denoise是另一類生成原語，多次重新生成不能等價補上。

優先順序由原先「常駐→pruned→DiT→Lite」修正為：**先以最新正式設定補warm做片證據→定位DiT瓶頸→針對pruned作有限相容性研究**。既有常駐現在就能用，不是新開發案。encoder小格式及LightVAE暫緩，除非新觀察顯示容量／I/O／decoder佔比確實值得優先處理。不再下載或刪除模型，也不清理Time Machine snapshots。

### 工具核查揭露的關鍵界線

1. **profile不能把wait當浪費、root-gpu當完整GPU時間。** `h3_gpu.h`／`h3_gpu.m`的root timestamp不涵蓋全部MPS child工作；direct int8 matmul不計入MPS linear次數，故`linear=0`不是沒有矩陣運算。`h3_gpu_profile_mark`的wall是兩次mark間隔，warm時可能夾入上一輪decode／輸出／使用者等待；後續驗收應量逐次`h3_generate`及正確phase邊界，不能直接挪用context lifetime total。
2. **已有bench，不先造平台。** `tests/bench_dit.c`及`h3_dit_bench`已有QKV／MLP／command等診斷，但固定fixture／shape／20步不等同最新768p／39f／RES-simple5 workload。先以代表性warm請求定位，必要時才做局部GPU timeline，不把強制fence擾動當正常成本。
3. **pruned不是綁固定步數的表。** [Comfy模型](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/minimax/model.py#L642-L729)以`t=1-sigma`映射`adaln_t_table`，在相鄰格點作線性插值；不是按第幾步查表。因此改steps／shift／simple本身不構成已證實的不相容。所選官方artifact的metadata／basis／數值誤差仍未核查，不能保證無損或速度。
4. **LoRA按target分類，不籠統否決。** `h3_lora.c`確實接受block/final AdaLN與time-embedder更新。只改未縮維attention／MLP等target的LoRA，沒有必然重建時間表的結構理由；改原始2688-wide AdaLN需相符basis與constant/bias轉換，改time embedder則原表失效。現行8-wide pruned-trained AdaLN另屬新增契約。任何不相容必須明確拒絕，不能仿效警告後跳過來假稱成功。[上游模型偵測](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/model_detection.py)、[官方格式說明](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)。
5. **encoder不是整份常駐GPU，pruned不等比例減warm RAM。** `h3_text_encoder.c`逐層讀取、釋放，保留的是conditioning；約4.5秒profile含GPU建立／載入／計算等，非純計算，也不含所有tokenizer／weight-store開啟成本。`h3_dit_schedule.c`原本已在prepare時計算AdaLN並逐block釋放大projection，所以少約26GBcheckpoint不能直接推算warm RAM同少26GB或每步同樣加速。
6. **LightVAE上限要看場景。** 完全消除13.9秒VAE最多約12.6%的說法只適用110.8秒cold-process樣本；warm分母與decoder load不同，須重新量測，不能拿它普遍否決Lite。

### 最多三項下一步（建議，未執行）

| 項目 | 最小範圍與驗收 | 停止門檻 |
|---|---|---|
| 最新正式warm做片驗收 | 同768p／39f／v1.2／BF16 context依序first、換seed、改prompt、改steps、回設定；記逐請求wall／core reload／conditioning／decoder／footprint。另選一個真實Ref2VA圖片＋音訊用途和一段目標片長，例如124f，完整看片聽音，不只解碼／hash | 用法、重載與品質問題足夠清楚即停，不擴全矩陣／daemon |
| DiT熱點定位 | 利用既有profile／bench測假說；必要時一個代表性warm forward的完整GPU timeline，區分SDPA、projection、MLP、搬移與提交空檔 | 沒有穩定熱點，或只有microbenchmark／dispatch數量／root timestamp優勢，就不落地kernel改動 |
| pruned BF16 target-aware研究 | 固定兩個task artifact，核對header/config/table及實際adapter是否更新時間支路；列支援／拒絕／未知，再決定數值探針。分開量磁碟、cold、warm reprepare、RAM | 為主要已接受targets需完整原模型或大量專用轉換且只剩不急的磁碟收益，就不升級為優先實作 |

Astra另指出兩個文件使用陷阱：Ref2VA immutable target刪完最後reference後，無reference request會轉FL2VA而產生target衝突；互動入口另建`/tmp/h3-XXXXXX`並清除原`output_path`，不能以原單次`-o`猜輸出位置。更換adapter／strength需新context。這些應作現有契約提醒，不當作缺少生成模式。

**本輪決定僅保存建議與證據；沒有執行上述warm／長片驗收、沒有改profile或導入pruned loader。** 日常定位仍是Mac原生主要H3生成引擎，沒有同條件ComfyUI對照，不宣稱效能冠軍或工作流等價。

## 收案決定

使用者於本輪同意暫時收案，先整理文件並commit／push，等待實際製作流程回饋。上列warm／長片驗收、DiT profile及pruned研究均是重開時的候選，不是現在排程。已有runtime與模型保留集合維持不變；不新增功能、不下載權重、不延伸刪除或清理APFS快照。