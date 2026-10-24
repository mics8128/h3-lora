# 加速設定研究

查閱日：2026-10-24。僅文件／來源研究，未執行推論、編譯、測試或格式化。

## 結論與來源界線

- 本次 **Civitai API 可讀取**：`lightx2v_4step_768p_v1.2` 是 version `3297123`、file `3181960`，SHA-256 `c8168ebc17bbacc4296103dda2fec1ba85b24392fa08cf2bfbcef0cff0dc3cc8`。頁面稱其為相容性轉換，並非上傳者自行訓練；不可將轉換頁的建議當成原作者受控實驗。[1]
- v1.1／v1.2 的頁面建議是 **5–8 steps、er_sde 或 res_multistep、simple、strength 1**。因此 v1.2 起點應為 **5 步**；檔名的 `4step` 不代表目前建議用四步。這是建議下限，不是引擎硬性禁止四步。頁面沒有給這兩版明確 shift；本地採用的 **12/3 是已揭露的預設值，不是已證實最優值**。[1][4]
- ModelTC 表格中的舊 **FL2VA 8-step v1.0（非 768p）**是 544p 混合比例訓練、shift **12/3**、蒸餾八步，但建議推論 **「8 / 4」**；舊 **4-step v1.0 768p** 則為 1344×768、shift **6/3**、推論四步。另有 **8-step v1.0 768p**，建議八步、6/3；不要把三者混用，也不要用舊版四步說明推翻 v1.2 的五步起點。[2]
- 查閱的 ModelTC README 沒有列 v1.2 規格。其 Studio 使用八步 768p v1.0，是部署選擇，**不是 v1.2 八步勝過五步的證據**。Civitai 的 5–8 範圍也沒有提供同 seed／同提示／同畫布／同採樣器的五對八步品質、音訊或時間數據。本地既有五步 22-frame 疊加 LoRA smoke 只證明該組合可產生短片，不能證明五步最優或八步必勝。[1][2][4]

## 544p 草稿與 768p 定稿

| 用途 | 權重／畫布 | 起點 | 注意 |
|---|---|---|---|
| 優先求快的草稿 | 舊非 768p FL2VA 8-step v1.0，file `3101211`；960×544 | 4 steps、Euler/simple、strength 1、12/3 | 四步是原作者「8 / 4」中較小值；不是 v1.2 降步。960×544 亦為上游 T2VA 工作流預設。[2][3][5] |
| v1.2 定稿起點 | file `3181960`；1344×768 | 5 steps、res_multistep/simple、strength 1、12/3 | 保持原生畫布；12/3 是本地 fallback。再與同版八步比較，不預先承諾品質提升。[1][4][5] |
| 需要原作者明確四步 768p 規格 | 舊 FL2VA 4-step v1.0 768p，file `3101219`；1344×768 | 4 steps、Euler/simple、strength 1、6/3 | 與 v1.2 是不同權重，不能據此推算 v1.2 四步品質。[2][3][5] |

544p 草稿換成 768p 定稿時同時換權重／畫布，不能視為構圖逐像素一致的升級，也不能用兩者時間隔離步數效益。若直接以 v1.2 跑 960×544，應標為跨解析度探索，不稱為該版已驗證最佳設定。

## 本地執行約束與小幅加速

- **simple 必須**：`--layers 50 --reuse 1 --core-reuse 1`、完整輸出畫布、關閉 token reduction；不得用內部縮圖後放大冒充原生 768p。限制不符合會在編碼前拒絕。原生 CFG 為 1；本地支援 Euler、LCM、RES，沒有 er_sde，所以 v1.2 選 `res_multistep`。[4][6]
- **row-fc2**：M5 的 `--use-int8-row-fc2` 是選用的數值近似，以每個 FC2 row 的 activation scale 搭配完整 TensorOps 乘積。本地 README 記錄完整 denoiser 約快 **2.6%**，並有兩支四步影片 SSIM 0.919／0.828；這不是 v1.2 768p 的端到端速度或無損品質保證。不可與 SSD streaming 併用，應最後單獨 A/B，不改步數、模型或其他精度。[6]
- **Base reuse 不等於 Turbo 蒸餾**：原本 Euler/serving 的 `--reuse 2/3` 在 20 步下分別只做 11／8 次全新 DiT 計算，其餘外推；`--core-reuse` 則減少昂貴核心更新。兩種非平凡 reuse 互斥；它們不是相同次數的完整模型推論，更不是四步 Turbo 等價物。simple 路徑不能套這些近似。[6]

## 最少有用試跑組（本研究未執行）

共同：固定 prompt、seed、輸入圖片與 base，單獨一個 Turbo LoRA、原生預設精度；50 layers、reuse/core-reuse 1、不縮內部畫布、不降 token；記錄時間、峰值記憶體、畫面連續性／主體／音畫同步。

1. **D4 草稿**：`3101211`，960×544，Euler/simple，4 步，12/3，strength 1。
2. **F5 定稿基準**：`3181960`，1344×768，RES/simple，5 步，12/3，strength 1。
3. **F8 步數對照**：完全複製 F5，只改 8 步；這才直接回答 v1.2 八步相較五步是否值得。
4. **F5-row（有 M5 才做）**：完全複製 F5，只開 row-fc2；隔離小幅加速與品質影響。

先用 22 frames 做短片篩查；對 F5/F8 的正式品質判斷採相同 **124 frames／24 fps（約 5.17 秒）**，完整觀看並聽音訊。短 smoke 不能取代完整時長對照；不得把 8/5 的步數比直接當端到端耗時比。[3][5]

## 引用與可重現性

[1] [Civitai model 2837571 API](https://civitai.com/api/v1/models/2837571)，本次可讀取；版本、檔案及頁面建議是可變資料。

[2] [ModelTC 原作者規格（固定 revision）](https://github.com/ModelTC/Minimax-H3-Turbo/blob/02e26d591f7a04d5d1a074c9566d5dd4f22f6225/README.md)；亦查閱 [目前 README](https://github.com/ModelTC/Minimax-H3-Turbo)。

[3] [原作者 ComfyUI 操作說明](https://github.com/ModelTC/Minimax-H3-Turbo/blob/02e26d591f7a04d5d1a074c9566d5dd4f22f6225/COMFYUI_SETUP_AND_INFERENCE.md)：Euler 預設、32 倍數畫布、24 fps 與 frame count。

[4] [本地 Turbo 設計與既有 smoke 紀錄](TURBO_DESIGN.md)。

[5] [本地比較規範](COMPARISON.md) 與 [2026-10-23 inventory](tools/turbo_models.json)：保留選定參數及權重 hash；若往後 API 不可用，應明示引用的是 **2026-10-23 快照**，不是當日完整 inventory。本次未重新擷取完整可持久化 API 快照。

[6] [本地 runtime reference：Turbo sampler selection／speed-quality preset](RUNTIME_REFERENCE.md)：simple、reuse、row-fc2 與 SSD 限制。歷史量測是文件紀錄，並非本研究重測結果。

## 本次一秒交付實測

推論主機：`ted`，Apple M5 Max、128 GiB；固定 seed 42、同提示詞、FL2VA base、單一 Turbo strength 1，未疊加 Unlocked。各組獨立 CLI 載入，依序各跑一次，非獨占或多次統計基準。所有組均實際完成所列 NFE、50 層、reuse/core-reuse 1、simple、shifts 12/3。

H3 原生時間形狀無法恰好輸出 24 幀：本次生成 **39 幀／24 fps（1.625 秒）**，保留 `native-39f.mp4`；取前 24 幀及音訊重編為 **`review-1s.mp4`（1 秒）**。下列時間均為完整 39 幀原生推論的實測，不是原生只生成一秒的時間，也未包含事後截片與複製。

| 成品資料夾 | 畫布 | Turbo / steps / sampler | 整體秒數（含載入） | 去噪秒數 |
|---|---|---|---:|---:|
| `01-draft-544p-v10-4step` | 960×544 | 舊非768p v1.0 / 4 / Euler | 90.638 | 14.333 |
| `02-draft-544p-v12-5step` | 960×544 | v1.2 / 5 / RES | 93.598 | 18.086 |
| `03-final-768p-v12-5step` | 1344×768 | v1.2 / 5 / RES | 148.968 | 50.722 |
| `04-final-768p-v12-8step` | 1344×768 | v1.2 / 8 / RES | 179.018 | 82.648 |
| `05-final-768p-v12-5step-rowfc2` | 1344×768 | v1.2 / 5 / RES，row-fc2 開啟 | 144.867 | 48.565 |

本機交付：`/Users/mics/Movies/H3-Acceleration-1s-20261024/`。各組包含原片、一秒片、0/12/23 幀原尺寸 PNG、五幀 contact sheet、engine/decode log、result.json；根目錄包含 summary.json 與 `ALL-5-COMPARISON.jpg`。本機所有遠端已記錄檔案 SHA-256 一致，五支一秒影片均在本機完整影音解碼通過；原片已於遠端完整影音解碼。抽樣影格已人工檢視，沒有主觀聽音或完整連續動作評分。

暫定選用：草稿採 01；若希望草稿延用正式版 v1.2 權重，採 02，但該解析度是跨解析度探索。正式先採 03；04 比 03 多 30.050 秒（約20.2%），抽樣畫面臉部、墨鏡與構圖有變化，不是像素一致的細節升級，尚不能宣稱較好。05 比 03 少 4.101 秒（約2.8%），抽樣構圖亦改變，不宣稱無損，因此不作正式預設。以上只是單一提示詞／seed 的候選篩選，不是全域最佳參數或長片品質保證。
