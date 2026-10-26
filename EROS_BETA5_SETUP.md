# Eros Max beta5：128G 原生配置

## 模型與執行位置

本配置使用作者提供的 **beta5 TURBO-hybrid 完整 BF16 檔案**，不是 beta6，也不是 Civitai 頁面的 W4A8 主檔。作者的 INT8 預設建議另見下表；不能把完整 BF16、ConvRot INT8 與 W4A8 當成相同表示。

| 項目 | 值 |
|---|---|
| 作者來源 | [TenStrip/10Eros-Max](https://huggingface.co/TenStrip/10Eros-Max/tree/main) |
| 原始檔 | `10Eros_Max_h3_TURBO-hybrid_beta5.safetensors` |
| 原始大小 | 40,222,982,192 bytes |
| 原始 SHA-256 | `098f138f3e899d03821dd8296c8db3db6fa16371e50fcdd38661d347fd9a1dfa` |
| 原生轉換 SHA-256 | `a61c17e55237c03f64c3db2620d11a09b5628fdf95fbdf7ac0f78c9dcc4f51bf` |
| 模型 root | `/Users/tedmacbookpro/projects/h3-models/Eros-Max-beta5` |
| 獨立 runtime | `/Users/tedmacbookpro/projects/h3-lora-eros-beta5` |
| 任務設定 | 模型 root 下的 `h3-task-profiles.json` |

原始檔保留在 `source/`，轉換紀錄為 `source/conversion-manifest.json`；實際推論使用 `FL2VA/transformer/model.safetensors`。`Ref2VA/transformer` 指向同一份 hybrid transformer。文字編碼器、tokenizer、影音 VAE 等共用官方 FL2VA 組件，**沒有以官方 transformer 的值補入作者權重**。初次安裝保持正式 `/Users/tedmacbookpro/projects/h3-lora` 不變；之後依使用者指示部署 review 修正，見本文末部署紀錄。

作者的[模型卡](https://civitai.red/api/v1/models/2851079)建議 beta5 預設使用 `TURBO-hybrid_int8`，其 [HF 檔案](https://huggingface.co/TenStrip/10Eros-Max/blob/main/10Eros_Max_h3_TURBO-hybrid_beta5_int8.safetensors) SHA 為 `4dd965496e5b1b83cd13c65cbe7a535b8a4d94ae768a7646b4e336d52c4781cf`。該格式有 per-row scale 與逆 Hadamard group rotation，不能用普通 `int8 × scale` 載入。本次選作者完整 BF16，避免假冒 ConvRot 支援；所有 preset 也明確停用原生 M5 的額外 INT8 MLP/QKV/attention-output 量化。這是精度／速度取捨，不是作者 INT8 工作流的速度或位元等價重現。

## 模型卡設定與 preset

[作者 README](https://huggingface.co/TenStrip/10Eros-Max/blob/main/README.md)與 Civitai 卡提供下列 beta5 起點，不構成每個場景的最佳品質保證。

| preset 中段名稱 | sampler / scheduler | 本配置步數 | 模型卡範圍／用途 |
|---|---|---:|---|
| `er-sde-beta57` | `er_sde / beta57` | 6 | 4–6；作者偏向風格保留、減少漂移 |
| `res-multistep-simple` | `res_multistep / simple` | 8 | HF 6–8；Civitai 6–9，偏運動品質 |
| `lcm-simple` | `lcm / simple` | 8 | 6–8；作者認為 Turbo 音訊較佳 |
| `euler-simple` | `euler / simple` | 6 | 4–8 |

共有八個明確設定：`eros-beta5-reference-<中段>-bf16-v1` 與 `eros-beta5-text-<中段>-bf16-v1`。reference 選 Ref2VA，text 選 FL2VA；兩者使用同一份作者 hybrid 權重。另已實作一般 `beta` scheduler，供作者 README 的 `er_sde/beta`、`lcm/beta` 組合使用。

共同設定：768×768、24 fps、50 層、denoise/core reuse 均為 1、無 token reduction、無 SSD streaming、F32 video VAE、三個 close-reference BF16 projection 選項開啟，**不疊加任何 Turbo 或風格 LoRA**。範例 107 幀約 4.458 秒；preset 上限 345 幀約 14.375 秒，遵守 `17k+5` 格點。preset 為 experimental，品質狀態不由載入成功自動升級。

- **Turbo 已融合進模型**：再疊官方 FL2VA／Ref2VA Turbo 不是本配置。
- 作者 README 要求參考生成不使用 cache／Spectrum。本配置停用改變 denoising 計算的 reuse／近似；不能把安全的不可變資產雜湊快取誤稱為模型近似 cache。
- video/audio shift 採 **12/3**，這是既有原生 H3 預設，**不是已查得的 beta5 專屬作者建議**。卡內明寫的 10/3 屬於 beta6，沒有照搬。
- 作者建議某些概念 LoRA 低強度疊加，不代表所有一般場景都需要；本次沒有下載、疊加或驗收這些概念 LoRA。
- 非 Turbo + PDD 的 6 warmup／4 PDD 替代流程不是本次 TURBO-hybrid 配置。PDD 需要 trunk adapter、分 interval 的影音 final-head bank、Euler 訓練格點與分階段切換；普通 LoRA 或改 sampler 無法等價替代。本版未宣稱支援。

## 使用

先進入獨立 runtime；既有一般模型的 sampler／scheduler 預設沒有改成 beta5 的 preset。

```sh
ssh ted
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
cd /Users/tedmacbookpro/projects/h3-lora-eros-beta5
export EROS_MODEL=/Users/tedmacbookpro/projects/h3-models/Eros-Max-beta5
mkdir -p outputs
./h3 --info -d "$EROS_MODEL"
```

純文字範例，不使用額外 LoRA：

```sh
H3_VAE_PRECISION=f32 ./h3 -d "$EROS_MODEL" \
  --sampler er_sde --scheduler beta57 --steps 6 \
  --video-shift 12 --audio-shift 3 \
  --width 768 --height 768 --frames 107 \
  --layers 50 --reuse 1 --core-reuse 1 \
  --use-slower-bf16-mlp --use-slower-bf16-qkv \
  --use-slower-bf16-attention-output --seed 42 \
  -p "A red fox walks across a snowy field. One continuous stable shot. Soft wind and footsteps, no dialogue or music." \
  -o outputs/eros-beta5-fox.mp4
```

參考生成仍採官方 Ref2VA 六段 prompt 與 `<Picture N>/<Audio N>` 標記。圖片和音訊用 `--ref-image`／`--ref-audio`；若還要指定目標開頭條件，用 `--guide-image 0 IMAGE`，不是把一般參考當成硬首幀。**Guide 不保證逐像素相同，音訊參考不保證音色／唇形或原樣拷貝**。不要用人物參考卻在 prompt 要求狐狸，然後把衝突結果當成首幀驗收。

列出 AI 任務 preset：

```sh
printf '{}\n' | python3 tools/h3_tasks.py profiles --json \
  --root task-state --config "$EROS_MODEL/h3-task-profiles.json"
```

依 [Runtime task interface](RUNTIME_REFERENCE.md#ai-task-interface) 完成 `asset.register → prepare → submit → worker → inspect`。預設 throughput，不產生可 exact-resume 的 checkpoint；需要 pause／resume 時，在 task 明確選 `execution_mode:"resumable"`。ER-SDE checkpoint 由同版 native 與任務介面共同識別，不能把其他模型或改過 prompt 的工作當成 exact resume。

## 原生補完與數值界線

完整版 beta5 仍是 **rank-8 AdaLN 曲線架構**，不是官方 full-width AdaLN。轉換器只讀官方 header 作名稱／形狀／dtype 契約，保留作者的表、basis、mean 與 rank-8 projection；52 個 QKV 矩陣轉為 native head-interleaved 排列，需要 F32 的 BF16 值僅做精確 bit widening。

原生執行 `t=1−sigma`、clamp 後在 1025×8 表上線性插值；curve 模式不套 pre-AdaLN SiLU。Block modulation 依 hidden dtype 使用 BF16；final 保留 **BF16 RMSNorm 邊界 → F32 scale/shift → F32 output heads**。專用 offset kernel 避免額外整段 hidden cast／slice，也不改一般官方模型的 BF16 final-head 快路徑。

`beta` 是 Beta(.6,.6)，`beta57` 是 Beta(.5,.7)，採 Comfy 的 999-index 格點、ties-to-even、去重與 terminal zero；去重後的實際 transition 數可能少於要求步數。ER-SDE 在 joint packed AV carried state 上運算，保留兩個 shift 的映射、歷史 clean prediction／derivative 與原生 RNG；first-sigma offset 用於模型計算，初始 noising 遵循 upstream 外層時序。ER checkpoint 使用 solver 3／revision 1／七個 float arrays，舊三個 sampler 保留原格式。**原生 Gaussian RNG 不與 PyTorch 同 seed 位元等價**，也不宣稱跨 backend 成品逐位相同。

## 驗證與審查

Astra 已執行設計及程式碼審查，找到並要求修正 final modulation 過早 BF16 rounding；修正後靜態 code gate 允許獨立整合實跑，並非品質推廣許可。

已觀察的 model-independent 結果：**5269 host checks**（包含單檔 Ref2VA 偵測的 failing-before／passing-after 回歸）、sampler suite、rank-8 projection／插值、實際 PyTorch 2.14.1 CPU oracle 對照的 16 與 **5376** 寬 Metal final-modulation、LoRA、GQA／group norm、AudioVAE primitives、VAE precision 全通過；converter 七個行為測試通過。最新 Python planner/asset suite **38 通過、2 個 inference opt-in tests skip**，另以真模型實跑下列 native／任務流程，不拿 skip 當成 inference 通過。

真正作者 beta5 權重已實跑：

| 路徑 | 實際條件與觀察 |
|---|---|
| 純文字 | 512×512／22幀／4步 Euler/simple；狐狸雪地成品可解碼且已看過影格 |
| 圖片＋音訊＋frame-zero guide | 512×512／22幀；ER-SDE/beta57 4步、LCM/simple 6步、RES/simple 6步、Euler/simple 4步皆生成並解碼成功 |
| ER exact resume | 真模型第1個 transition 暫停後恢復；clean latent 與 MP4 SHA 均與完整未中斷工作相同；原生 checkpoint 也成功由 task asset registry 註冊 |
| 768 preset 尺度 | 768×768／107幀／6步 ER-SDE/beta57，約4.458秒影片；一次完整wall **315.240秒**。成品已看過影格，參考人物與公園可辨識並有揮手動作；不是逐像素或音色驗收 |
| 已安裝 AI preset | 八個profile均以完整dependency hashes回報available且成功prepare；實際reference ER profile透過prepare/submit/worker/inspect完成pause→resume→succeeded，發布media與clean latent |

起初參考請求因「只認 index 的可選 Ref2VA 偵測」回傳 `missing_model`。已改為共用真實目錄 inventory：單檔／分片均可，缺少或空可選目錄不影響純文字，損壞／非目錄／讀取失敗則拒絕；不是補一個假的 index。原失敗、修正後成功及各次 binary SHA 都保留於 `acceptance/summary.json` 與 `acceptance/failures/ref-index-only/`。

修正後 native executable SHA-256 為 `071b4309c658e0a18002ef3a11a31f90dbbe68d45a14e4315c5df66a3e3daea2`。原生 manifests／JSONL／diagnostics／成品位於獨立runtime的 `acceptance/`；任務流程紀錄在 `task-acceptance/`。本機影片、來源／轉換紀錄與摘要保存在 `/Users/mics/Movies/H3-Eros-Beta5-20261026/`。另一次原模型 serving-Euler 成品 SHA 與舊基準相同，但不是 beta5 的速度比較。

以上時間是單次、process-cold且filesystem cache未清除的實跑，不是普遍吞吐量或品質排名。畫質、語音身分、唇形、長片穩定性與所有參數組合仍需各自判斷；不能把 [先前官方模型驗證](RUNTIME_REFERENCE.md#evidence-boundary-for-this-integration)改稱 beta5 成績。

## 來源與授權

- [作者 HF README](https://huggingface.co/TenStrip/10Eros-Max/blob/main/README.md)、[檔案與 SHA API](https://huggingface.co/api/models/TenStrip/10Eros-Max/tree/main?recursive=true)、[beta5 version API](https://civitai.red/api/v1/model-versions/3294059)、[模型卡 API](https://civitai.red/api/v1/models/2851079)。
- [Comfy H3 曲線／final layer，固定 commit](https://github.com/Comfy-Org/ComfyUI/blob/fe298e4e20f1ff26d30e8639267bbd0b57811390/comfy/ldm/minimax/model.py)。
- [Comfy ER-SDE](https://github.com/Comfy-Org/ComfyUI/blob/fe298e4e20f1ff26d30e8639267bbd0b57811390/comfy/k_diffusion/sampling.py)、[beta scheduler](https://github.com/Comfy-Org/ComfyUI/blob/fe298e4e20f1ff26d30e8639267bbd0b57811390/comfy/samplers.py)、[CONST sampling](https://github.com/Comfy-Org/ComfyUI/blob/fe298e4e20f1ff26d30e8639267bbd0b57811390/comfy/model_sampling.py)。
- [`beta57` 的具名綁定](https://github.com/ClownsharkBatwing/RES4LYF/blob/56e44755d866c87ff9cb4b6608c8e42155db6ecc/sigmas.py#L1420-L1421)。
- [Comfy-kitchen ConvRot](https://github.com/Comfy-Org/comfy-kitchen/tree/f8b7e97a1418d2e529bd5405fbf86312be198ae6)、[PDD 工作流](https://github.com/Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc)。
- [原始 ER-SDE 論文／MIT solver](https://github.com/QinpengCui/ER-SDE-Solver/tree/25ca6c2e6b065754a55694dc6941a3259fa0f082)；完整 notice 與 Comfy GPLv3／RES4LYF AGPLv3＋額外商用限制的區分見 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。參考數值契約不是授權重授、clean-room 證明或法律清查。

作者聲明適用 MiniMax H3 community license，且移入 LTX 2.3、Wan 2.2、Krea 2 的部分還受各來源條款約束。Civitai 的 allow 欄位不是整體商用或再分發權利的法律保證；本次沒有接受額外法律條款或宣告授權已完整審核。

## 128G review 更新 (2026-10-26)

依使用者指示，已推送並部署 review 修正 `0f96a9bd859f1576097327e2b7f46298b28d31d8`。主目錄 `/Users/tedmacbookpro/projects/h3-lora` 由 upstream 基準加舊工作區改動更新為 origin/main；beta5 runtime 從同一提交匯出來源。兩處均強制重新編譯 `h3`、`libh3.a`、host 與 sampler tests。後續文件提交不改變這次驗證的程式來源。

部署前確認沒有執行中的生成程序。原主目錄與 beta5 runtime 完整 APFS clone 備份在 `/Users/tedmacbookpro/projects/h3-deploy-backup-20261026-0f96a9b/main` 與 `beta5`；任務設定亦備份在同一目錄。主目錄原本未提交的 tracked/untracked 改動另保留為 Git stash `pre-deploy-0f96a9b`。模型、既有輸出與八個 preset 參數未更動。

部署後兩個實際入口各通過 5,293 host checks、sampler tests、beta/beta57 effective schedule 查詢，並以作者 beta5 執行圖片＋音訊 reference、image guide、兩步 ER-SDE/beta57 的 256×256／22-frame 生成。兩處 latent SHA-256 均為 `30a41f2aa23a6db12274801870f3839fe80ebb4ec7cb43f9f1aa34b8d66d89aa`，MP4 均為 `6c7102c015f5b4e31085af7e0dcac31991886dc38b52759a21ebaca69adb808a`，與 review acceptance 一致；FFmpeg 完整解碼通過。已安裝 task profile 查詢回報八個 preset 全部 available。

主目錄執行檔 SHA-256：`11168f5316f6027e933fbd99c14d202e5b6fec1f91a7dcafb163b19808efe847`；beta5：`e5a88771f646cfc4bcc01b87011255ae10a14c78c47055b92fd37af2ba8689ab`。各 runtime 的 `outputs/deploy-0f96a9b/` 保存 manifest、JSONL、stderr、latent、MP4 與 summary；beta5 `runtime-source.json` 區分本次部署與舊版歷史 acceptance。

舊 checkpoint／已準備計畫綁定舊版執行身分，不應跨版本直接恢復或提交。新工作重新 prepare；需要原版計算狀態時，先確認無工作執行，再回復對應完整 runtime 備份至原路徑，不要只換單一 executable。備份路徑直接啟動不等於已驗證的 exact resume。
