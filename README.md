# h3-lora / h3-metal

Apple Silicon 原生 MiniMax-H3 影音生成：文字／首尾幀走 **FL2VA**，有序圖片、影片、音訊參考走 **Ref2VA**。支援 immutable 多 LoRA、Turbo sampler、互動常駐，以及可選 BF16 video decoder。

這份文件只放**日常用法**。完整 C API、所有速度近似、診斷與測試見 [RUNTIME_REFERENCE.md](RUNTIME_REFERENCE.md)；目前模型、共用依賴及已完成清理見 [MODEL_INVENTORY.md](MODEL_INVENTORY.md)。

模型清理：已依指示刪除4個額外風格LoRA與Eros轉換前／後權重；官方FL2VA／Ref2VA、**21個Turbo全部**及Unlocked均保留。Eros資料夾只剩紀錄與共享連結，不再是可用模型root。

## 階段收案狀態

2026-10-25：**目前階段暫時收案，功能凍結，等實際製作流程回饋。** 已發布的runtime基線為`e791b32`（前一階段`a2cfb36`）；本次只整理文件、模型清理紀錄與Astra建議，不新增功能或效能宣稱。

主要生成入口、Turbo、互動常駐及BF16選項已有相應實跑證據；這不代表完整ComfyUI相容、長片品質全面驗收或同機效能領先。pruned、LightVAE、encoder新格式、kernel優化與進階生成控制均暫緩，不自動啟動benchmark、下載模型或進一步清理。

先用下列命令做實際工作。只有出現具體品質問題、等待瓶頸或無法表達的必要操作，再以該場景重開；記錄prompt、參考、seed、設定、first／warm耗時及成品，不要求先做全矩陣。詳細候選見[Astra唯讀核查](ASTRA_OPTIMIZATION_PLAN.md#2026-10-25綜合應用與下一階段建議astra唯讀核查)，屬後續參考而非本次待辦。

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

v1.2 檔名含 `4step`，但其卡片建議起點是 **5步**。草稿v1.0與正式v1.2是不同權重；同seed不代表兩者構圖一致。想只保留v1.2也可用960×544做草稿，但屬跨解析度探索，不稱為原作者最佳設定。詳細來源與早期A/B見 [ACCELERATION_RESEARCH.md](ACCELERATION_RESEARCH.md)。

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

以下使用現有驗收素材；自己的素材換掉兩個路徑及prompt。圖片順序會成為`<Picture 1>`、`<Picture 2>`，不是由檔名判定人物／商品。

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

FL2VA Turbo不能拿來替代Ref2VA Turbo。`--first-frame/--last-frame`是FL2VA首尾幀條件，不是一般Ref2VA參考，兩組入口不能混用。純音訊及混合參考已有實際接線，但不保證音色、節奏、唇形或人物一致性；影片原聲是conditioning，不是原樣拷貝至輸出。[完整參考入口與限制](RUNTIME_REFERENCE.md#8-add-image-video-and-audio-references)。

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
| 接續一個鏡頭 | 抽前片尾幀作下一次FL2VA first-frame，或把前片作Ref2VA參考 | 可做接續流程，但不是共享latent的無縫延續，誤差可能累積 |
| 任意時間點的多個guide／局部遮罩重繪／ControlNet | 目前不能以幾次CLI等價替代 | 需要引擎實作相應conditioning、mask或控制模型 |
| latent放大後partial-denoise refine | 目前沒有對應CLI契約 | 需要latent輸入／輸出、起始噪聲與schedule續接，不能把重新生成稱為refine |

一般多步流程是編排問題；任意時間點guide、mask、ControlNet與latent refine才是新增生成原語。ComfyUI workflow JSON／custom nodes不能直接由本程式執行，也不宣稱輸出逐像素相同。需要完整進階控制時仍可選ComfyUI；不必為一般批次與剪接先建立一套節點框架。

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
