# ComfyUI 如何減少 H3 重複載入成本

## 範圍與可重現來源

2026-10-24 透過 [GitHub commits API](https://api.github.com/repos/Comfy-Org/ComfyUI/commits/7cebca52227c63b4180ceed448cbd7ad22903b6d) 固定 Comfy-Org/ComfyUI **`7cebca52227c63b4180ceed448cbd7ad22903b6d`**（commit 時間 2026-10-24T02:22:35Z）。以下為官方程式碼閱讀，不是 Mac 上的 ComfyUI/H3 效能測試；沒有執行 inference、build 或 tests。自訂節點可能有自己的 `IS_CHANGED`、輸入依賴與載入策略，不能直接套用標準節點結論。

## 不是「全部永遠留在 GPU」，而是四個不同層次

1. **節點物件持續存在**：`objects` 以 `(node_id, class_type)` 作 key；執行時先找既有物件，找不到才建構。這讓 `LoraLoader.loaded_lora` 等物件內狀態可跨 queue 使用。這不等於該節點輸出的模型仍在 GPU。[CacheKeySetID](https://github.com/Comfy-Org/ComfyUI/blob/7cebca52227c63b4180ceed448cbd7ad22903b6d/comfy_execution/caching.py#L68-L81)、[物件取得](https://github.com/Comfy-Org/ComfyUI/blob/7cebca52227c63b4180ceed448cbd7ad22903b6d/execution.py#L493-L502)。
2. **節點輸出快取**：`outputs` key 包含 class、`IS_CHANGED` 結果、字面輸入及所有上游輸入簽章；命中直接回傳，不執行節點。模型 loader 的輸出可保留 Python 模型／patcher 物件；文字編碼節點則可保留 conditioning。這是跨任務結果重用，不是少做 diffusion steps。[簽章](https://github.com/Comfy-Org/ComfyUI/blob/7cebca52227c63b4180ceed448cbd7ad22903b6d/comfy_execution/caching.py#L83-L149)、[命中短路](https://github.com/Comfy-Org/ComfyUI/blob/7cebca52227c63b4180ceed448cbd7ad22903b6d/execution.py#L440-L452)。
3. **裝置載入／offload 管理**：`current_loaded_models` 尋找已載入 patcher；`free_memory` 根據所需空間卸載其他模型，必要時可 partial unload。CPU 模型物件保留、權重重新移到運算裝置，和重新從檔案反序列化，是不同成本。[model_unload](https://github.com/Comfy-Org/ComfyUI/blob/7cebca52227c63b4180ceed448cbd7ad22903b6d/comfy/model_management.py#L812-L848)、[free_memory / load_models_gpu](https://github.com/Comfy-Org/ComfyUI/blob/7cebca52227c63b4180ceed448cbd7ad22903b6d/comfy/model_management.py#L893-L1038)。
4. **LoRA 描述與實體權重分離**：先建立 patcher／patch 描述，之後才依載入路徑 materialize；細節如下。不能將它描述成每次調 strength 都免費，也不能描述成每次都重讀 base checkpoint。

## 改哪些輸入，會重做哪些工作？

前提：同一 ComfyUI process、相關輸出尚未被淘汰、標準有向圖且沒有額外失效條件。下表是依輸入依賴推導的行為，不是實測耗時。

| 修改 | 可重用 | 需要重做／注意 |
|---|---|---|
| **只改 sampler seed** | 上游模型 loader、LoRA loader 輸出、未改的文字 conditioning | seed 所在節點與其下游失效；通常重新採樣、VAE decode、儲存，不重編碼同一文字。若 seed 也接到其他上游節點，範圍跟著圖變化。 |
| **只改 prompt 文字** | 獨立的 base 模型／LoRA loader 輸出 | 對應文字編碼與其下游採樣、decode 失效；文字編碼器可能須從 CPU 重新移到運算裝置，這不等於從磁碟重讀 checkpoint。 |
| **改 LoRA strength** | base loader 輸出；同一 LoraLoader 物件中同一路徑的已讀 LoRA tensor | LoRA 節點輸出及相依下游失效；建立新 patch 設定並可能 unpatch/repatch。若 CLIP 經過此 LoRA 節點，文字編碼也會因圖上祖先輸入改變而失效，**即使只調 model strength、clip strength 沒變**；獨立 CLIP 分支則不受影響。 |

上表依據前述祖先簽章與短路程式碼；`LoraLoader` 同一路徑直接取 `self.loaded_lora`，只有未命中才 `load_torch_file`，且 model/clip strength 都為零會直接回傳原物件。[nodes.py:L714–759](https://github.com/Comfy-Org/ComfyUI/blob/7cebca52227c63b4180ceed448cbd7ad22903b6d/nodes.py#L714-L759)。

## LoRA：clone / add_patches 不等於立即合併所有權重

- `comfy.sd.load_lora_for_models` 做 key mapping、LoRA 轉換／解析，然後 `model.clone()`、`add_patches(loaded, strength_model)`；CLIP 類似。因此只調 strength 可省 LoRA 檔案重讀，但函式內轉換、mapping、patch 建立並非全部被省略。[sd.py:L110–139](https://github.com/Comfy-Org/ComfyUI/blob/7cebca52227c63b4180ceed448cbd7ad22903b6d/comfy/sd.py#L110-L139)。
- 一般 `ModelPatcher` 的 `get_clone_model_override` 回傳同一底層 model 與 backup；`clone` 複製 patch 列表／設定，不默認 deep-copy 整份 checkpoint。**特殊 dynamic／force-deepcopy 分支例外**。[model_patcher.py:L432–463](https://github.com/Comfy-Org/ComfyUI/blob/7cebca52227c63b4180ceed448cbd7ad22903b6d/comfy/model_patcher.py#L432-L463)。
- `add_patches` 對合法 key 記錄 `(strength_patch, patch, strength_model, offset, function)`，更新 UUID。`patch_weight_to_device` 才保存 base backup、建立計算 dtype 暫存、呼叫 `comfy.lora.calculate_weight`、round/copy/set 實際權重。[add_patches](https://github.com/Comfy-Org/ComfyUI/blob/7cebca52227c63b4180ceed448cbd7ad22903b6d/comfy/model_patcher.py#L847-L885)、[materialization](https://github.com/Comfy-Org/ComfyUI/blob/7cebca52227c63b4180ceed448cbd7ad22903b6d/comfy/model_patcher.py#L922-L951)。
- 一般完整載入會呼叫上述 materialization；low-VRAM 路徑可掛 `LowVramPatch` 延至權重使用時處理，或因強制 patch／shape 改變而先做。不能一概宣稱「永遠 lazy」或「永遠預先 merge」。[load](https://github.com/Comfy-Org/ComfyUI/blob/7cebca52227c63b4180ceed448cbd7ad22903b6d/comfy/model_patcher.py#L1005-L1106)。
- patch UUID 改變時，一般 `partially_load` 會還原 backup 再載入／套 patch；`unpatch_model` 可還原原權重、移至 offload device、清理 loaded 標記。loader 在共享 model 的 clone 間切換時亦有 detach。這可有顯著記憶體流量，不可視為零成本。[unpatch](https://github.com/Comfy-Org/ComfyUI/blob/7cebca52227c63b4180ceed448cbd7ad22903b6d/comfy/model_patcher.py#L1153-L1192)、[UUID 判斷](https://github.com/Comfy-Org/ComfyUI/blob/7cebca52227c63b4180ceed448cbd7ad22903b6d/comfy/model_patcher.py#L1279-L1305)、[clone detach](https://github.com/Comfy-Org/ComfyUI/blob/7cebca52227c63b4180ceed448cbd7ad22903b6d/comfy/model_management.py#L981-L989)。此 revision 另有 `ModelPatcherDynamic` override，不能把一般 patcher 的完整載入細節套用所有後端：[dynamic load](https://github.com/Comfy-Org/ComfyUI/blob/7cebca52227c63b4180ceed448cbd7ad22903b6d/comfy/model_patcher.py#L1945-L1979)、[dynamic partially_load](https://github.com/Comfy-Org/ComfyUI/blob/7cebca52227c63b4180ceed448cbd7ad22903b6d/comfy/model_patcher.py#L2228-L2246)。

## Cache 與記憶體 flags 的界線

- 此 revision 啟動預設是 **RAM-pressure cache**，不是一律 classic：active headroom 預設系統 RAM 10%（至少 2、最多 10 GiB），inactive 預設系統 RAM（最多 128 GiB）。`--cache-classic` 用舊式積極清除策略；`--cache-lru N` 保留最多 N 個節點結果；`--cache-none` 同時將 outputs、objects 設為 NullCache，因此連 LoraLoader 物件內的檔案快取也不能依上述跨次物件重用假設。RAM-pressure 會依壓力淘汰，不是永久保留。[main.py:L339–357](https://github.com/Comfy-Org/ComfyUI/blob/7cebca52227c63b4180ceed448cbd7ad22903b6d/main.py#L339-L357)、[CacheSet](https://github.com/Comfy-Org/ComfyUI/blob/7cebca52227c63b4180ceed448cbd7ad22903b6d/execution.py#L121-L154)、[RAMPressureCache](https://github.com/Comfy-Org/ComfyUI/blob/7cebca52227c63b4180ceed448cbd7ad22903b6d/comfy_execution/caching.py#L523-L564)。
- `--disable-smart-memory` 控制**模型 offload**，不是關閉節點結果 cache：`free_memory` 對可卸載模型改用很大的釋放要求，而非只補足所需裝置空間。通常增加積極 offload，不能用它來追求「權重永遠留 GPU」。[flag](https://github.com/Comfy-Org/ComfyUI/blob/7cebca52227c63b4180ceed448cbd7ad22903b6d/comfy/cli_args.py#L194)、[實際分支](https://github.com/Comfy-Org/ComfyUI/blob/7cebca52227c63b4180ceed448cbd7ad22903b6d/comfy/model_management.py#L893-L924)。
- `--highvram` 與 `--gpu-only` 的說明偏 GPU 常駐，但 **MPS 初始化最後會覆蓋為 `VRAMState.SHARED`**，因此不能把 CUDA 的 HIGH_VRAM 行為直接搬到 Mac。[state 覆寫](https://github.com/Comfy-Org/ComfyUI/blob/7cebca52227c63b4180ceed448cbd7ad22903b6d/comfy/model_management.py#L573-L595)。
- `--gpu-only` 仍有獨立作用：text encoder 的執行／offload device、VAE offload device、intermediate device 回傳 `get_torch_device()`；MPS 時就是 `mps`。沒有此旗標時這些 offload／intermediate 預設可在 CPU；`--highvram` 不等價於這些分支。統一記憶體不代表 PyTorch CPU/MPS 張量、轉換、額外暫存都沒有成本，也不保證模型和採樣工作區一起放得下。[device](https://github.com/Comfy-Org/ComfyUI/blob/7cebca52227c63b4180ceed448cbd7ad22903b6d/comfy/model_management.py#L195-L204)、[text/intermediate/VAE 策略](https://github.com/Comfy-Org/ComfyUI/blob/7cebca52227c63b4180ceed448cbd7ad22903b6d/comfy/model_management.py#L1213-L1288)。這些 flags 不改模型架構、不減少 diffusion steps；本研究沒有其 Mac 加速倍率證據。

## 對 h3-lora 的保守對照（工作樹快照）

- 本地 `h3.c:33–90` 已有 context 級 conditioning、prepared DiT、video decoder cache 與 clear/enable API；`main.c:590–631` 同一 context 進入互動 CLI。只重啟單次 CLI process 無法跨 process 延續這些物件，這與 ComfyUI 常駐 process 不同。
- **只改 seed**：`h3_conditioning_key`（`h3.c:141–164`）及 `h3_prepared_key`（`166–192`）不包含 seed；在其他 key 不變且 cache 開啟時可重用。不是 ComfyUI 專有能力。
- **改 prompt 的具體差異**：本地 prepared key 把 conditioning（含 prompt）、shape、steps 等綁在一起；`h3.c:1052–1057` key 改變就釋放整個 `ctx->dit`。ComfyUI 圖中獨立 diffusion loader 不依賴 prompt，因此改文字不必使 loader 輸出失效。可借鏡的是**將不變的 DiT 權重／準備狀態和會變的 conditioning、shape 工作區分開失效**；目前程式碼不代表這種解耦已完成，也沒有其實測速度數據。
- 本地 LoRA 在 `h3_load_dir`（`h3.c:482–513`）建立 stack，`h3_lora.c:h3_lora_load_target` 負責帶 LoRA 的 target 載入；不能假設已有 ComfyUI 的可變-strength patcher、base backup 與 clone 切換生命週期。借鏡「保留 base／LoRA 原料、避免重讀檔案」不等於直接套用 Python clone，也不保證省去 merge／重建量化權重成本。
- ComfyUI cache 是**避免跨請求重做**；h3-lora 的少 steps、layer pruning、token reduction、core reuse 則是**減少單次採樣工作量或近似計算**。兩類方法可分開討論，不能把節點 cache 當成每個 diffusion step 的近似 cache，也不能據此推算合併倍率。

本節本地來源為調查當下工作樹，其他開發可能同步變更；上述具體函式名比行號更適合後續定位。未改引擎、未跑效能測試。

## 單次生成：精度與 VAE 路線

- [Comfy H3 官方指南](https://docs.comfy.org/tutorials/video/minimax/minimax-h3) 與 [Comfy-Org 模型包](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/README.md) 提供預先量化的 int8_convrot／fp8_scaled／w6a8 checkpoint，與完整／pruned BF16 是不同儲存及運算契約。量化檔可減少讀取資料量，但不能假設 LoRA 套入量化模型完全不需反量化、patch 或重新量化。ConvRot 含旋轉基底，不能當成現有 h3-lora 原生 int8 buffer 直接讀取。
- 官方指南記載 int8 video VAE 約2× decode，但不是這台 Mac 的測量；原生 NVIDIA 快路徑、Comfy Kitchen、Sage 與 sparse attention 有硬體及CUDA限制。MPS不存在相同kernel時可能fallback，不應承諾倍率。
- [ModelTC LightVAE](https://github.com/ModelTC/LightVAE) 的 H3 Lite 採**block pruning**，保留原始 latent interface；[Comfy 整合](https://github.com/ModelTC/LightVAE/blob/main/comfyui/README.md) 可獨立替換 encoder／decoder，音訊VAE不變，FL2V範例保留原始首尾幀encoder。H3 BF16路徑不需要自訂Triton，因此架構上比移植CUDA kernel更值得評估；但其公開測量是H200，不是M5。
- LightVAE 的 ConvRot-INT8 decoder 另需相符載入器；native NVIDIA建議CUDA13+。本地 `h3_video_vae.c` 固定36層，載入並運算的是F32（大型linear已有MPS F32路徑），不是原生BF16 VAE；不能把Lite或ConvRot檔案換路徑就直接使用。改用BF16也屬數值策略變更，須另外驗證。Lite還需正確讀取保留層／metadata並實際驗證重建畫面。
- `match` 參考圖縮放、低解析度草稿、Turbo縮步可減少單次工作，但有內容與畫質取捨。Comfy的latent upscale＋refine仍需高解析度去噪與VAE decode，不等於便宜的Lanczos影片放大。

## 已有本地時間基準與待驗證項目

既有768p／39幀／v1.2五步單次CLI觀測：Qwen4.402s、DiT load/preparation51.035s、denoise50.722s、video VAE39.991s、audio VAE0.387s、整體148.968s。DiT load包含讀權重、LoRA合併、量化與預計算，不可把51秒全歸因SSD。各細項未單獨隔離計時。

這些是既有 h3-lora 真實執行紀錄，**不是ComfyUI同機比較**。單次優化應量測cold-process完整wall及上述各階段；常駐另測同prompt換seed、改prompt、改steps/shape的cache命中與失效。不能把完全不變workflow直接回傳舊節點輸出當成新的生成速度。使用方式維持既有CLI／互動入口，優先自動重用，不引入cache層級或後端旗標讓使用者逐一設定。

## Astra 指導後實際採用的方案

完整提案及依量測調整的裁定見 [ASTRA_OPTIMIZATION_PLAN.md](ASTRA_OPTIMIZATION_PLAN.md)。本輪沒有照搬 Comfy 的 graph framework，也沒有新增 daemon、全域 model registry、磁碟 merged-checkpoint cache 或使用者 cache flags。

1. **單次冷啟動**：large LoRA target在原128-row／64-rank算術下以最多兩個有界worker處理不相交rows；同一block的FC1／FC2／QKV／out使用原quantizer，但合併為一次同步submit。沒有改steps、解析度、層數、量化精度或開啟row-fc2。
2. **常駐與request分離**：context保留單一resident DiT/GPU owner；完整request key仍包含prompt、shape、steps等，resident key只保留mode／adapter／precision等身分。request改變會重新refine text、建立layout/maps/RoPE/workspace及condition rows，而不是將舊prompt錯當cache hit。
3. **AdaLN快取**：只有兩個sigma clocks與condition-modality presence相同才沿用；改steps/shifts或增減音訊modality會重算。full50、無SSD／token reduction、sequence政策相容才重準備，其餘採完整重載。
4. **互動caller**：改references／anchors不再直接全清cache，由owned輸入簽章失效。`!cache clear`及結束context仍明確釋放模型；LoRA snapshot維持immutable契約。`-p` one-shot結束後不可能保留RAM供另一程序使用。

最初A-factor預解碼cache沒有證明cold wall收益，已撤除；四個真實large target的全部merged bytes曾與原panel路徑一致，因此不把它誤寫為後續數值不一致的已證實根因。MPS graph cache以shape/dtype/operation建graph並以placeholder重綁當前tensor，未發現舊activation/weight buffer的地址索引洩漏；不同shape的graph entries仍會保留至GPU teardown，沒有宣稱任意長session的graph記憶體完全固定。

## 驗收中揭露並修正的GPU同步競態

Ref2VA圖片＋音訊的fresh/warm比較最初失敗。取消於Qwen文字編碼完成後的快探針顯示：五次fresh／clear／shape／back的video/audio原始condition rows相同，唯文字embedding不同；獨立vision encode重複六次的merged/deepstack bytes相同。

根因定位到 `h3_gqa_causal_bf16`：讀取共享`reductions[0]`的softmax maximum後，其他SIMD group尚未讀完，先完成者已將同一scratch覆寫為sum。獨立CPU softmax oracle在168 tokens捕獲 **0.746718** 最大誤差。`h3_vae_encoder_group_norm_silu_f32`的mean→variance也有同一種讀寫競態。兩處均補上threadgroup barrier，而非抑制錯誤或放寬數值門檻。

新增fixture-free `h3_reduction_tests`：32／33／168／257 tokens的causal GQA對double softmax oracle，及多SIMD GroupNorm/SiLU對獨立CPU oracle，每組重複24次並比較相同輸入的完整GPU output bytes。修正前GQA oracle失敗，修正後所有oracle及repeatability通過。完整conditioning快探針修正後五次text/video/audio bytes一致。一般host1768 checks、LoRA數值／snapshot／量化／Metal projection（含新並行loader完整bytes及讀取失敗）、sampler tests也已通過。

最終冷對照的新舊executable使用**同一份修正後shader**，避免把競態修正造成的舊錯誤輸出差異當作量化取捨或加速效果。模型/adapter/設定固定，新的diffusion仍需執行，不以舊成品快取冒充新生成。

## 最終同機量測與交付

Apple M5 Max128GiB；四次交錯順序為舊→新→新→舊，各次獨立程序，**沒有清空OS page cache**，不是獨占機器或大量統計。舊新版executable都使用同一份已修正競態的shader。FL2VA、Lightx v1.2強度1、RES/simple5步、shifts12/3、50層、reuse/core-reuse1、seed42、相同prompt與1344×768／39幀。時間是完整39幀推論含載入，不含另截一秒及傳輸；不是只生成24幀的時間。

| 實測項目 | 舊版 | 優化版 |
|---|---:|---:|
| 冷程序整體wall第1次 | 149.247秒 | 139.820秒 |
| 冷程序整體wall第2次 | 146.357秒 | 135.500秒 |
| 冷程序整體wall中位數（各2次） | 147.802秒 | 137.660秒 |
| DiT load／prepare中位數 | 49.942秒 | 37.259秒 |
| process peak physical footprint範圍 | 25.150–25.151GB | 25.166–25.168GB |

這組冷wall少10.141秒，約**6.86%**；DiT準備少約25.4%。沒有宣稱整體2倍或把暖快取效果算入單次收益。新舊四支原片的完整decoded RGB SHA-256均為 `27e3ca6ff8a4ea3658fffa8b017814d66612f24f8876d8e5c314a64a1f829ca1`，PCM SHA-256均為 `812ba910965dd935e763ddd0b554285d140219be518369d02cfb4f95099f85f0`。去噪與VAE時間仍有主機波動，VAE本輪沒有更換精度／架構。

常駐另用512×512／22幀／同五步adapter的同一context依序測，不可和上表直接比較：

| 狀態 | 修改前單次觀察 | 最終版單次觀察 |
|---|---:|---:|
| 初次生成 | 65.096秒 | 51.314秒 |
| 改prompt | 64.539秒 | 15.128秒 |
| 只改seed | 8.860秒 | 8.813秒 |
| 改steps至6 | 58.695秒 | 19.081秒 |
| 明確清cache後生成 | 67.048秒 | 47.608秒 |

改prompt、seed及steps均沒有重新load core；clear後實際重載一次。最終resident probe peak physical footprint為37.761GB，常駐不是零記憶體成本；深度多shape的fresh/warm矩陣觀察FL2VA38.132GB、Ref2VA圖片＋音訊40.702GB、無LoRA36.304GB，都是process footprint，不與RSS相加。

**完整實際驗收**：FL2VA9個warm transition對5組unique-fresh oracles、Ref2VA圖片＋音訊11個warm transition對6組unique-fresh oracles、無LoRA9個warm transition對5組unique-fresh oracles，全都通過RGB callback全bytes雜湊／完整decoded PCM雜湊、metadata與core reload語義檢查。涵蓋不同token長度prompt、seed、steps/back、shape/back、clear、取消/recover，以及Ref2VA音訊移除/還原。不是用只看cache log、沒throw或source文字斷言代替行為驗證。

**真實互動CLI**也實跑：Ref2VA初次62.30秒、加第二張商品圖26.27秒、移除後19.54秒、明確clear後60.70秒。加圖與移除均重用core；移除及clear後原設定的RGB/PCM與第一支一致，雙圖的輸出確有改變。這個smoke使用8步Ref2VA adapter；數值cache矩陣的5步是等價性測試，不是該權重的正式品質建議。

**本機交付**：`/Users/mics/Movies/H3-Optimization-20261024/`。188個遠端記錄檔案已逐一核對SHA-256；包含117支原片／預覽（保留探索過程與失敗證據，不能把舊失敗目錄當成最終版），所有117支在這台Mac完整影音解碼成功。最終以 `final-verification.json` 的 `all_success=true`、`fixed-cache-*`、`*-corrected-cold-*`、`fixed-final-resident` 和 `interactive-ref-edit` 為準；`FINAL-QUALITY-OVERVIEW.jpg`為抽樣畫面總覽。未作主觀音訊、唇形同步或長片品質評分。

## LightVAE 是否值得做成選項

**值得做獨立、明確選用的草稿 decoder，但不作預設，也不在這輪 BF16 移植時混入。**標準 H3 Lite 是 block pruning：原36層保留12層 `[1,3,6,9,13,16,18,21,26,27,30,34]`，不是將36層任意減成前12層。[官方 config](https://github.com/ModelTC/LightVAE/blob/main/lightvae/config/h3-lightvae.json) 保留24channel latent interface、normalization、tile256/overlap64及各架構參數；原生 loader 必須驗證實際checkpoint key/index remapping，不能把缺層當成零權重或直接換檔名。

[官方模型庫](https://huggingface.co/lightx2v/LightVAE/tree/main)列出標準 Lite decoder約3.25GB、encoder約406MB，另有約834MB的ConvRot-INT8 decoder。第一個有意義的選項應只替換**標準 Lite decoder**，沿用原始encoder與audio VAE。[官方 ComfyUI 整合](https://github.com/ModelTC/LightVAE/blob/main/comfyui/README.md)要求FL2V首尾幀條件仍用原始encoder；Ref2V可分別替換encoder/decoder，但本專案不因此擴大到encoder替換。ConvRot-INT8是不同checkpoint及運算契約，不等同於現有Metal int8 primitive，本輪不移植。

[官方專案](https://github.com/ModelTC/LightVAE)報H3解碼最高2.97×及原始overall VBench-2.0分數99.51%，測量背景是H200，不是M5，也不是端到端生成倍數。**[INFERENCE]** 只有decoder佔用時間能被這項改動縮短，DiT去噪、文字編碼及LoRA載入仍存在；即使解碼快三倍，也不能宣稱影片生成快三倍。BF16和裁層是正交取捨，應先比較完整F32 vs Lite-F32，再比較各自BF16，不能將兩項差異歸因給一項。

落地前置條件：固定權重revision與SHA256、確認原始block index／緊密重編index、dtype與normalization一致、所有tensor shapes匹配，並驗證單tile／空間接縫／時間chunk重疊／resident preview及FL2VA/Ref2VA。模型庫的Apache-2.0標籤不應被解讀為覆蓋H3衍生權重的所有限制；採用及再散布前須另查[H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)及適用地區／用途。沒有把模型權重納入git或自動下載。

本輪結論是**列為下一個草稿選項候選**，不是已實作LightVAE；沒有本機LightVAE速度／畫質證據。原始完整decoder繼續作正式品質基準，BF16選項不需要LightVAE權重。

## PR75／PR73 移植後的最終量測

已實作PR75式per-GPU-context lazy direct pipeline；library仍啟動時編譯，不改shader算術或thread-safety契約。已獨立移植PR73的BF16大型projection／SDPA，保留F32預設與F32norm／RoPE／SwiGLU／residual／小投影。`H3_VAE_PRECISION=bf16`明確選用；只接受`f32`／`bf16`，未設等同`f32`，空值／其他值直接錯誤。decoder key加入有效precision，故precision切換只重建decoder而不重載相容DiT。BF16模式不配置用不到的F32attention-output scratch。沒有移植disk cache／prefetch／int8 VAE，也沒有變更tile範圍或引入LightVAE權重。

### 同一最終executable、同prompt／seed、1344×768／39幀

M5 Max128GiB，Lightx v1.2強度1、RES/simple5步、50層、reuse/core-reuse1；完整推論含載入與原片輸出，不含一秒剪輯／傳輸。最終順序F32→BF16→BF16→F32，各2次新process，沒有清OS／Metal caches，沒有聲稱整機獨占。39幀是1.625秒；另提供一秒預覽，不能把整體時間當作只生成24幀。

| 指標 | 最終F32 | 最終BF16 |
|---|---:|---:|
| 完整wall第1次 | 138.685秒 | 114.170秒 |
| 完整wall第2次 | 133.596秒 | 107.458秒 |
| 完整wall中位數 | 136.140秒 | 110.814秒 |
| video VAE profile中位數 | 40.868秒 | 13.909秒 |
| process peak physical footprint範圍 | 25.165GB | 25.165–25.933GB |

這組完整wall少**18.60%／25.327秒**，VAE階段約**2.94×**。最小32×32／5幀的direct decoder smoke則F32約1.09秒、BF16約1.33秒，**小畫布不保證變快**；不要以大型畫布收益外推所有情況。

PR75獨立比較的已發布eager基線兩次137.587／134.460秒，中位數136.023秒；最終lazy F32中位數136.140秒，差約0.117秒，屬本次波動尺度，**沒有重現作者約14%的收益**。先前lazy F32樣本中位數135.667秒亦只差0.356秒。保留lazy機制是為避免未用pipeline建置及刪除固定kernel清單，不宣稱本機顯著加速。最終四次F32／BF16及基線的音訊PCM全相同；F32完整decodedRGB與基線相同，BF16兩次完整decodedRGB相同但不等於F32。

### 品質與實際接線驗證

- 公園人物、雪地狐狸各一組768p／39幀F32/BF16原片，附左右一秒比較；抽樣檢視composition與主體接近，沒有宣稱長片品質／唇形／所有紋理無損。公園初次對照與最終對應precision的完整RGB/PCM雜湊相同；狐狸只有初次一組對照。
- 完整解碼H.264RGB8比較PSNR：人物37.200dB、狐狸41.179dB；最大channel差51／52levels。**包含codec效應，不是decoder floatPSNR**，不能拿它和PR73作者的float75.3dB直接比較或宣稱最大只差4–6levels。
- 另用固定**合成latent**直接比較未編碼floatRGB：32×32／5幀63.644dB、最大0.003001；352×256／39幀62.238dB、最大0.029974。這不是語意／人物品質樣本，也沒有重現作者75.3dB。352寬確實跨空間tile，39幀跨時間chunk。
- F32與BF16各自驗證ordinary chunked／resident final完整floatRGB byte-identical；resident preview與ordinary對應中間幀byte-identical。也實跑32×32／5幀direct diagnostic入口，並非只測CLI的單一路徑。
- 真實FL2VA及Ref2VA圖片＋音訊context各跑F32→BF16→F32→unset。只初次load core一次，後續全部0次；回F32／unset的完整callbackRGB雜湊與第一支一致，BF16不同，排除沿用錯precision decoder。Ref2VA使用8步相符adapter；這個256平方smoke驗接線，不當作768p品質基準。空值與`int8`precision均實際拒絕。
- fixture-free `h3_vae_precision_tests`納入`make test`，獨立literal oracle驗BF16 widening後加F32bias、row broadcast、dtype／容量／overflow邊界。最終build無警告，該測試、host1768checks、LoRA numeric/snapshot/Metal、sampler、多SIMD reduction檢查全通過；沒有宣稱執行缺少fixtures的整套MLXparity。

### 成品與重現

本機`/Users/mics/Movies/H3-Lazy-BF16-20261024/`。最終驗收看`final/verification.json`的`all_success=true`、`final/summary.json`及兩個resident目錄的smoke.log。根目錄summary／原片是初次完整人物／狐狸對照，`final/`是去除unused scratch後的最終binary重複量測；保留兩者，不混淆binarySHA。`park-f32-left-bf16-right-1s.mp4`、`fox-f32-left-bf16-right-1s.mp4`與`F32-vs-BF16-QUALITY.jpg`可直接觀看。沒有將影片、模型或本機benchmark harness提交到git；原始driver作成品重現附件。
