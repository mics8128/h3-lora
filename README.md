# h3-lora / h3-metal

Apple Silicon 原生 MiniMax-H3 影音生成：FL2VA／Ref2VA、immutable 多 LoRA、Turbo、互動常駐與可選 BF16 video decoder。開發版另接入參考＋時間定位 guide、native latent／mask／partial denoise、checkpoint 續跑，以及供小模型使用的結構化任務工具。

這份文件只放**日常用法**。完整 C API、所有速度近似、診斷與測試見 [RUNTIME_REFERENCE.md](RUNTIME_REFERENCE.md)；目前模型、共用依賴及已完成清理見 [MODEL_INVENTORY.md](MODEL_INVENTORY.md)。

模型清理：先前依指示刪除4個額外風格LoRA與 **beta6** 轉換前／後權重；官方FL2VA／Ref2VA、**21個Turbo全部**及Unlocked均保留。beta6資料夾仍只剩紀錄。另依新指示安裝作者完整 **Eros Max beta5 BF16**，使用獨立runtime與明確preset，見 [beta5配置](EROS_BETA5_SETUP.md)。

## 開發狀態與品質界線

2026-10-26：依實際製作回饋重開開發。Astra 設計見 [DEVELOPMENT_DESIGN.md](DEVELOPMENT_DESIGN.md)；實作不等於已驗收人物一致性、音色、唇形或無縫接縫。既有128G部署沒有自動更新；新程式在獨立工作目錄驗證，不覆蓋正在使用的原始碼或中斷生成。

原有 Turbo／resident／BF16 有先前短片證據；新增條件與取樣狀態另作驗收。開發版已接入 beta5 的 rank-8 AdaLN 曲線與 ER-SDE／beta／beta57；這不代表任意 pruned 格式、LightVAE、ConvRot／NVFP4、Sage／TeaCache及官方未開源 Context-IR／2K 都可用。查核來源見 [加速工作流審計](ACCELERATION_WORKFLOW_AUDIT.md)、[品質搭配研究](QUALITY_PROFILE_RESEARCH.md) 與 [beta5配置／驗證](EROS_BETA5_SETUP.md)。沒有同機同條件證據，不能宣稱「7秒1080p／205秒」、「幾乎無損」或「完美續拍」。

完整 fork review 已以原始 upstream `8974cc0` 為基準，修正 CLI 首尾影格、guide 音訊與相鄰判定、checkpoint 身分／前置驗證，以及 task 準備／修訂邊界；獨立 beta5 實跑的 uninterrupted/resume latent 與 MP4 hashes 相同。範圍、測試與未驗證界線見 [review 紀錄](RUNTIME_REFERENCE.md#fork-wide-review-corrections-2026-10-26)。既有部署未被替換。

## 0. 小模型主要入口：JSON任務，不拼長指令

`tools/h3_tasks.py` 使用 Python 標準函式庫；FFmpeg／FFprobe用於素材探測與明示的媒體處理。管理者先設定 `H3_TASK_ROOT`（可寫任務目錄）、`H3_TASK_CONFIG`（引擎、device、已安裝模型／adapter與完整dependency hashes的設定檔）。所有共用同一GPU的state roots必須使用相同device ID。

```sh
printf '{}\n' | python3 tools/h3_tasks.py profiles --json \
  --root "$H3_TASK_ROOT" --config "$H3_TASK_CONFIG"
```

小模型只需選回傳的profile ID、註冊素材取得asset ID，再依 `prepare → submit → inspect` 使用。`submit`不自行啟動GPU工作；管理者或編排程序執行 `worker`，以一個device一次一個工作處理佇列。`revise`是修改後重生成，`continue`是生成接續內容，`resume`才是恢復原計算；三者不可互換。

`standard_media_v1`將常規裁切／重採樣／音訊補齊列入準備計畫，提交digest表示採用該精確計畫；內容截斷、改台詞、丟條件或降低內部畫布不會靜默執行。完整schema、設定與可執行命令見 [任務介面](RUNTIME_REFERENCE.md#ai-task-interface)。


## 1. 執行位置與設定

目前分工：這台 Mac 管理原始碼／文件／Git 及觀看成品；編譯、模型檢查、測試與生成均在 `ted`。權重不放進 Git，也不要把影片或合併權重提交進來。

先登入遠端，再設定以下變數；路徑是目前部署，不是程式硬編碼限制：

```sh
ssh ted
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
cd /Users/tedmacbookpro/projects/h3-lora
export H3_MODELS=/Users/tedmacbookpro/projects/h3-models
export MODEL_DIR="$H3_MODELS/MiniMax-H3"
export REF_MODEL_DIR="$H3_MODELS/MiniMax-H3-Ref2VA"
export DRAFT_LORA="$H3_MODELS/loras-turbo/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
export FINAL_LORA="$H3_MODELS/loras-turbo/minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors"
export REF_LORA="$H3_MODELS/loras-turbo/minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
make -j4
mkdir -p outputs
./h3 --info -d "$MODEL_DIR"
./h3 --info -d "$REF_MODEL_DIR"
```

`--info` 不生成影片；不帶 LoRA 時不載入其 payload。Ref2VA root 仍需 loader 要求的 FL2VA/shared inventory，**不能因只做 Ref2VA 就刪掉 FL2VA 樹**；實際連結見模型盤點。

## 2. 選哪一組

以下是有來源與短片實測的**起點**，不是所有內容的最佳參數，也不保證長片品質。預設不疊加風格 LoRA。

| 用途 | 模型／LoRA | 畫布 | 步數／sampler | video VAE |
|---|---|---|---|---|
| 草稿 | FL2VA＋舊非768p v1.0 | 960×544 | 4／Euler | F32；BF16可獨立試 |
| 正式起點 | FL2VA＋v1.2 768p | 1344×768 | 5／RES multistep | F32預設 |
| 正式加速候選 | 同正式起點 | 1344×768 | 5／RES multistep | 明確選BF16，接受數值差異 |
| 人物／商品／影片／音訊參考 | Ref2VA＋相符v1.0 8-step | 1344×768 | 8／RES multistep | F32預設；BF16可選 |

共同使用 `simple`、strength 1、50 layers、reuse/core-reuse 1、shifts 12/3。12/3 是已揭露的採用預設，不是每版作者證實的最優值。`simple` **不能**搭配 reduced layers、reuse>1、token reduction 或較小 internal canvas；要降低草稿解析度，直接改輸出 `--width/--height`。

v1.2 檔名含 `4step`；本頁保留的5步RES／12–3是既有本機配方，不是已證明最佳的官方設定。LightX作者v1.2發佈建議為4步Euler／6–3；兩組應分開比較，不能將FL2VA的6–3套到Ref2VA。草稿v1.0與v1.2也不同，同seed不保證構圖一致。來源與品質比較設計見 [QUALITY_PROFILE_RESEARCH.md](QUALITY_PROFILE_RESEARCH.md)。

### 草稿：先看構圖與動作

```sh
H3_VAE_PRECISION=f32 ./h3 -d "$MODEL_DIR" \
  --lora "$DRAFT_LORA" 1 --lora-model fl2va \
  --sampler euler --scheduler simple --video-shift 12 --audio-shift 3 \
  --width 960 --height 544 --frames 39 --steps 4 \
  --layers 50 --reuse 1 --core-reuse 1 \
  --seed 42 -p "A red fox walks through snow. Stable camera, natural light." \
  -o outputs/draft.mp4
```

### 正式：同一題材的768p起點

```sh
H3_VAE_PRECISION=f32 ./h3 -d "$MODEL_DIR" \
  --lora "$FINAL_LORA" 1 --lora-model fl2va \
  --sampler res_multistep --scheduler simple --video-shift 12 --audio-shift 3 \
  --width 1344 --height 768 --frames 39 --steps 5 \
  --layers 50 --reuse 1 --core-reuse 1 \
  --seed 42 -p "A red fox walks through snow. Stable camera, natural light." \
  -o outputs/final.mp4
```

要試較快的VAE，將上例 `H3_VAE_PRECISION=f32` 改成 `H3_VAE_PRECISION=bf16`，並用不同輸出檔名保留對照。只接受 `f32`／`bf16`；未設定等同F32，空值／其他值拒絕。BF16不需要新權重，**不是LightVAE**，也不是無損模式；不改音訊VAE或參考encoder。

最終M5 Max短片量測：同1344×768／39幀／5步，各2次，完整生成中位數F32 **136.140秒**、BF16 **110.814秒**，約少18.6%；VAE階段40.868→13.909秒。這是特定短片，不是所有工作負載倍率；小畫布可能不變快。完整條件、品質差異與快取驗收見 [COMFY_CACHE_RESEARCH.md](COMFY_CACHE_RESEARCH.md#pr75pr73-移植後的最終量測)。

### Ref2VA：先選正確base與adapter

以下是既有39幀接線短測，不是正式品質驗收；官方輸出範圍4–15秒，品質比較可用124幀（約5.17秒）。自己的素材需換掉路徑與prompt。圖片順序成為`<Picture 1>`、`<Picture 2>`；人物與聲音關係應按官方格式明確定義，不由檔名推定。

```sh
H3_VAE_PRECISION=f32 ./h3 -d "$REF_MODEL_DIR" \
  --lora "$REF_LORA" 1 --lora-model ref2va \
  --sampler res_multistep --scheduler simple --video-shift 12 --audio-shift 3 \
  --width 1344 --height 768 --frames 39 --steps 8 \
  --layers 50 --reuse 1 --core-reuse 1 --ref-image-size match \
  --ref-image outputs/ref-lora-wiring-20261024/inputs/person.png \
  --ref-audio outputs/ref-lora-wiring-20261024/inputs/reference.wav \
  --seed 42 -p "The fully clothed adult in Picture 1 walks and waves in a sunny park." \
  -o outputs/reference.mp4
```

FL2VA Turbo不能替代Ref2VA Turbo。開發版將首尾圖統一為時間定位guide；有ordered references時選Ref2VA，可同時帶guide，但必須使用相符Ref2VA adapter。guide是固定條件，不保證輸出像素相同；參考聲音也不保證音色、節奏或唇形，原聲不是原樣拷貝。[完整參考與任務介面](RUNTIME_REFERENCE.md#ai-task-interface)。

### 2026-10-26：main 整合與 128G 核對

- 本機將 `feature/turbo-comparison` 快轉整合至 `main`，功能基準為 `7f87de2`；未改動生成程式。
- 128G（`ted`）部署目錄 `/Users/tedmacbookpro/projects/h3-lora` 的 87 個受版控檔案，逐檔 SHA-256 與此功能基準一致。遠端 Git 仍是 upstream `8974cc0` 加未提交部署檔，**檔案一致不代表遠端 Git 已同步 main**；本次未切換分支、覆蓋檔案或重啟工作。
- 遠端 `make -q h3` 回傳 0，表示依現有 Makefile 與檔案時間無待重建項目；這不是執行檔可重現建置的雜湊證明。
- 在遠端以既有 Ref2VA root，同時傳入 `--ref-image`、`--ref-audio`、`--first-frame`，512×512／22幀／2步，實際 CLI 回傳 exit 1：`h3: full references cannot be combined with frame anchors`。請求在參數驗證階段拒絕，未生成影片。
- **該次核對的舊部署不能三項同用。** 後續開發版已加入references＋guides，不能把本段舊部署拒絕紀錄當成新程式或模型本質的限制。反之，開發版可編碼也不代表舊部署已更新或首幀／聲音品質已保證。
- 核對當下遠端有 video-maker 的 Ref2VA 圖／音工作執行中；未中斷、未另啟完整推論。本次驗證的是三項組合確實被拒絕，不是重新驗收圖／音生成品質。


## 3. 連續改prompt：用互動常駐

在對應命令中省略 `-p` 與其prompt，其他模型／sampler設定保留，即進互動模式。不需要新增cache flags或啟動daemon。互動模式會另建啟動畫面顯示的`/tmp/h3-XXXXXX`目錄，不沿用單次命令的`-o`；要保留成品請依顯示路徑複製，或用`!save PATH`保存。

- `!status` 看當前設定；`!help` 看命令。
- `!seed random` 換seed；直接輸入新prompt繼續生成。
- Ref2VA中用 `!ref-image PATH`、`!ref-remove N`、`!refs` 管理有序圖片。
- Ref2VA immutable target若移除最後一個reference，先加回參考再生成；無reference的請求會走FL2VA，與該context target不符。要純文字生成，另開FL2VA context。
- `!cache clear`釋放模型、decoder與工作快取；immutable adapter snapshot留至context結束。
- 更換LoRA檔案／強度／target需重新啟動context。單次`-p`結束後不會替下一個process保留RAM。

改prompt／steps／相容shape／references會重建必要request狀態，保留相容權重；不相容精度policy、streaming或近似模式可能重載。這不是跳過新的diffusion，也不是回傳舊影片。

## 4. 一秒預覽與複製回Mac

H3是24fps，支援的原生幀數為`5+17*n`：22幀約0.917秒、39幀1.625秒，**不能原生恰好24幀**。上述短測先跑39幀，再截成一秒；生成時間仍是完整39幀的成本。

在`ted`產生預覽：

```sh
ffmpeg -i outputs/final.mp4 -t 1 \
  -c:v libx264 -crf 16 -c:a aac outputs/final-review-1s.mp4
```

在這台Mac另開終端執行：

```sh
mkdir -p ~/Movies/H3-Review
rsync -av ted:/Users/tedmacbookpro/projects/h3-lora/outputs/final.mp4 \
  ted:/Users/tedmacbookpro/projects/h3-lora/outputs/final-review-1s.mp4 \
  ~/Movies/H3-Review/
```

既有可看片目錄（Mac）：

- `~/Movies/H3-Lazy-BF16-20261024/`：最新F32/BF16原片、左右一秒比較、最終驗收與source archive。
- `~/Movies/H3-Optimization-20261024/`：冷啟動／常駐及Ref2VA互動驗收；以`final-verification.json`和`fixed-*`為準。
- `~/Movies/H3-Acceleration-1s-20261024/`：較早的草稿／正式／8步／row-fc2候選比較，不當作最新binary速度。


## 5. 綜合應用：先用指令串接，不必重建節點系統

本專案定位是Mac原生H3生成引擎；多步製片可以由shell／Python分次呼叫CLI，再由FFmpeg剪接、轉碼或混音。**不需要ComfyUI節點圖，才有資格做綜合應用。** 下表區分流程編排與引擎能力，不把一般組合使用全都列成「進階功能尚缺」。

| 要做的事 | 現在可怎麼組合 | 界線 |
|---|---|---|
| 多個prompt／seed批次挑片 | 分次CLI生成、記錄輸出；固定模型可用互動常駐減少重載 | 多個獨立process不共用RAM快取 |
| 人物＋商品＋動作／聲音參考 | 同一次Ref2VA命令重複reference flags即可 | 不必為每個參考拆一次；不保證身份或口型品質 |
| 草稿→正式 | 先低解析度篩選，再用正式設定重新生成 | 不是同latent接續refine；換權重／畫布可能改構圖 |
| 多鏡頭成片、字幕、配樂、外部放大 | 分別產生片段，再用FFmpeg或其他製片工具組合 | 外部處理，不代表H3引擎內有對應節點；外部工具另需具備該能力 |
| 接續一個鏡頭 | 任務`continue`選bounded tail guide，明示overlap與join policy | 新片段生成，不是checkpoint resume；不保證無縫，拼接是明示後製 |
| 任意時間點的圖片／短片／音訊guide | native guides與任務`prepare` | 條件定位，不能冒充輸出latent或原音硬鎖 |
| 同畫布latent局部去噪 | native clean latent＋同形狀mask＋start_transition；任務`revise` | 0保留／1生成，保留的是latent而非逐像素畫面；不是通用像素遮罩編輯器 |
| 中斷後恢復計算 | `pause`／`resume`與完整sampler checkpoint | 需相同依賴／profile／prompt／schedule；不是只存latent |
| ControlNet／latent放大refine／官方2K | 尚無對應已接妥契約 | 不能用guide、重新生成或像素resize冒稱完成 |

ComfyUI workflow JSON／custom nodes不能直接執行；本機任務規格只表達引擎真實能力，不建立任意節點平台，也不宣稱與Comfy逐像素等價。

## 6. 效能定位與下一步

目前已有原生Metal、Turbo、常駐權重、LoRA準備優化及可選BF16 VAE的實際證據，適合作為本機短片生成工具；**沒有同機、同模型／精度／輸入的ComfyUI對照，不能宣稱比ComfyUI快或已是Mac最快。** 短測速度也不能外推完整5–15秒或大型多參考任務。

以下只是在實際流程發現瓶頸時的候選順序，**本階段不啟動新工作**；已完成的常駐現在即可使用。

1. **先看真實流程**：用現有互動context減少重載，記最新正式設定的first／warm耗時與成品品質；需要支援更長片時再驗對應目標片長。
2. **有證據才定位DiT**：沿用既有profile／bench；`wait`包含GPU工作、root timestamp不含全部MPS child工作，不能只看數字名稱就認定attention或同步是瓶頸。
3. **pruned BF16只列有限相容性研究**：核對實際artifact與adapter targets；時間表會插值，不能因改steps／shift就一概否決，也不能承諾小檔等比例更快。

encoder小格式與LightVAE暫緩；若之後有明確容量／I/O／decoder佔比問題再重排。Lite收益須分cold與warm情境，不能把cold樣本的上限當通用結論。

最新BF16兩次39幀短測的phase中位數：DiT準備36.455秒、去噪53.154秒、video VAE13.909秒；準備包含讀取、合併、量化及預計算，不能全算成SSD。各phase中位數相加不必等於整體中位數。來源與驗證條件見[量測記錄](COMFY_CACHE_RESEARCH.md#效能判斷與剩餘優化空間)。

## 7. 文件怎麼看

| 需求 | 文件 |
|---|---|
| 日常生成 | 本README |
| 現有模型、真實容量、共用連結與已完成清理 | [MODEL_INVENTORY.md](MODEL_INVENTORY.md) |
| 所有CLI/API、無LoRA base/serving模式、首尾幀、診斷、編譯與測試 | [RUNTIME_REFERENCE.md](RUNTIME_REFERENCE.md) |
| 草稿／正式adapter來源與早期短測 | [ACCELERATION_RESEARCH.md](ACCELERATION_RESEARCH.md) |
| 已落地常駐／BF16、最新量測與LightVAE評估 | [COMFY_CACHE_RESEARCH.md](COMFY_CACHE_RESEARCH.md) |
| Turbo全檔重現套件與21權重manifest | [COMPARISON.md](COMPARISON.md)、[tools/turbo_models.json](tools/turbo_models.json) |
| 歷史設計與Astra後續建議，非自動待辦清單 | [LORA_DESIGN.md](LORA_DESIGN.md)、[TURBO_DESIGN.md](TURBO_DESIGN.md)、[ASTRA_OPTIMIZATION_PLAN.md](ASTRA_OPTIMIZATION_PLAN.md) |
| Ref2VA來源與實際支援範圍 | [REF_COMPATIBILITY_RESEARCH.md](REF_COMPATIBILITY_RESEARCH.md) |
| 上游PR取捨與落地狀態 | [UPSTREAM_RESEARCH.md](UPSTREAM_RESEARCH.md) |
| 程式與第三方授權 | [LICENSE](LICENSE)、[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) |

Unlocked v2.1等風格adapter屬選用實驗，不是日常必需；兩張卡片的sampler建議可能衝突，見[已記錄的組合測試](TURBO_DESIGN.md#combined-768p-smoke)。LightVAE尚未實作，不要下載後直接替換原始VAE；模型權重授權和MIT程式碼授權分開檢查。
