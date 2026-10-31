# Eros Max beta7：128G 原生配置

## 模型與位置

| 項目 | 值 |
|---|---|
| 作者來源 | [TenStrip/10Eros-Max](https://huggingface.co/TenStrip/10Eros-Max/tree/main) |
| 原始檔 | `10Eros_Max_h3_8stepTURBO_beta7.safetensors`（BF16） |
| 原始大小／SHA-256 | 40,222,934,448 bytes／`b1361691a953ff7623e5dc628a76e338d592da12e3c9d5652cecb7eeaaa4aa94` |
| 原生轉換 SHA-256 | `1a6e97fbf5041c29514fb891c4f88f61211553218b885337fe6c4a1e06e32eae` |
| 模型 root | `/Users/tedmacbookpro/projects/h3-models/Eros-Max-beta7` |
| 獨立 runtime | `/Users/tedmacbookpro/projects/h3-lora-eros-beta7` |

beta7 與 beta5 同為 rank-8 AdaLN 曲線格式（`adaln_t_table [1025,8]`、每層 `[*,8]` projection），差別只在**沒有** `adaln_basis`／`adaln_mean`。這兩個張量 Comfy 與 h3.c 都不讀，所以舊轉換器拒收 beta7 純屬過嚴。`tools/convert_eros.py` 現在接受已知的 beta5／beta7 雜湊、basis/mean 可省略、`adaln_t_table` 必備；beta5 輸出位元組不變。原始檔頭部 metadata 為 FastH3 Omni timestep、`qrusher_gates_included=no`（無 VSA gate），dense forward 即 Comfy 的標準路徑。

INT8 版（`beta7_int8HQ`、`raw_beta7_int8`）是 ConvRot per-row 格式，原生 runtime 不支援，未使用。

```sh
cd /Users/tedmacbookpro/projects/h3-models/Eros-Max-beta7
python3 /Users/tedmacbookpro/projects/h3-lora-eros-beta7/tools/convert_eros.py \
  --source source/10Eros_Max_h3_8stepTURBO_beta7.safetensors \
  --schema source/official-transformer.header.json \
  --output FL2VA/transformer/model.safetensors > source/conversion-manifest.json
```

`FL2VA/` 其餘組件 symlink 至官方 FL2VA；`Ref2VA/` 全部 symlink 至 `FL2VA/`（作者稱 beta7 為完整 hybrid，一份權重跑所有模式）。QKV 以 52 個矩陣做 band→head 交錯重排；來源每 128 列 RMS 統計呈三段分明（Q／K／V 連續），交錯假設則均勻，確認來源是 Comfy 連續排列。

## 作者建議與本配置

作者 README：shift **12/3**，8stepTurbo 版「6–12 步」，起點 **Euler/simple 8 步**或 **er_sde/sgm_uniform**。本次新增原生 `--scheduler sgm_uniform`（Comfy `normal_scheduler(sgm=True)`，見 [RUNTIME_REFERENCE](RUNTIME_REFERENCE.md#turbo-sampler-selection)）。Turbo 已內建，**不要再疊 Turbo LoRA**。作者也指出單次生成易有背景閃爍，Comfy 以 2-pass／3D latent upscaler 處理；原生尚未實作該 upscaler。

```sh
ssh ted
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
cd /Users/tedmacbookpro/projects/h3-lora-eros-beta7
./h3 -d /Users/tedmacbookpro/projects/h3-models/Eros-Max-beta7 \
  --sampler er_sde --scheduler sgm_uniform --steps 8 \
  --video-shift 12 --audio-shift 3 --layers 50 --reuse 1 --core-reuse 1 \
  --width 768 --height 768 --frames 107 --seed 42 \
  -p "A red fox walks across a snowy field. One continuous stable shot." -o outputs/fox.mp4
```

video-maker 改用 beta7：profile 的 `engine` 指向 beta7 runtime、`ref2va_model` 指向 beta7 模型 root，`scheduler` 設 `sgm_uniform`、`steps` 8。beta5 runtime 舊執行檔不認得 `sgm_uniform`。

## 實跑驗收（2026-10-31，M5 Max 128G，process-cold）

| 案例 | 條件 | wall | 觀察 |
|---|---|---:|---|
| 純文字 | 512²／22幀／Euler simple 8 | 27 s | 狐狸雪地，影格清楚連貫 |
| 純文字 | 512²／22幀／ER-SDE sgm_uniform 8 | 29 s | 同上 |
| 參考＋音訊＋首幀 guide | 768²／107幀／Euler simple 8，F32 VAE、BF16 投影 | 370 s | 人物、服裝、公園與揮手保留 |
| 同上 | ER-SDE sgm_uniform 8 | 367 s | 同上 |
| beta5 對照（10/26） | 同輸入／seed，ER-SDE beta57 6 | 315 s | — |

四支 MP4 均完整解碼。右上樹葉區塊的相鄰影格亮度差（ffmpeg `signalstats` YDIF 平均）：beta7 Euler **11.1**、beta7 ER/sgm **7.4**、beta5 **3.5**，與作者描述的背景閃爍一致；ER-SDE/sgm_uniform 明顯較低，建議作為 beta7 預設起點。這是單一 seed 的量測，不是品質或音色認證。

成品、manifest、JSONL 與 log 位於 runtime 的 `acceptance/`。

## 公式查證（2026-10-31）

針對「撕裂感」逐段比對 Comfy master `7c4390a6`：
- **權重轉換**：532 個張量逐位元組相符，其中 QKV 52 個、F32 擴寬 112 個、其餘 368 個。
- **sampler／scheduler**：用相同注入噪聲跑完整 8 步迴圈，video／audio 誤差 ≤3.6e-7。
- **AdaLN 曲線與 rank8 投影**：用實際權重比對，誤差為 0。
- **RoPE、位置、QKV／SwiGLU 順序**：都相符。
- **DiT 內 BF16 中間捨入**：與 Comfy eager 有差異，原生在 F32 做融合運算，不是公式錯誤。

這輪查證確認 VAE 拼接錯誤：tile 交叉點原本與未混合的鄰塊混合，而且在混合前就先 clamp，已依 Comfy 修正，合成結果誤差 ≤3.6e-7。時間閃爍指標不變，未解決使用者後續回報的人臉直線。

beta7 前約 20 幀的高變化量，在 latent 階段就已存在：前 5 個轉換的 RMS 為 0.589，同設定 beta5 為 0.475。這表示差異不只出在最終 VAE 解碼；尚不能單憑此數據排除生成實作差異。

**未對齊**：
- mask 生成的逐 token 時間步（Comfy #15375）。
- ER-SDE 初始音訊 packing 約 2e-5 的偏差。
- σ=0 端點的時間步差異，約 2e-6。

## VAE 解碼幾何對照（2026-11-01）

同一個 768²／107 幀 latent，分別以原生及隨模型提供的官方 PyTorch VAE 解碼。官方設定使用 256px tile，原生舊 auto 選 288px；tile 大小同時改變 attention 上下文和正規化 RoPE 位置，不是無損效能參數。

- 以同樣 H.264 CRF18 編碼後，原生 288px 相對官方 PSNR 為 30.54dB，256px 為 38.32dB；第26、30幀的人臉 crop 在256px較接近官方。
- 官方 raw 相對自身 CRF18 為36.94dB；上述數值不能當作 raw tensor parity 或已到編碼誤差下限的證明。
- 預設已固定256px。無環境覆寫的實跑輸出，RGB逐位元組等於先前明確設定 `H3_VAE_TILE_PIXELS=256` 的影片。
- `H3_VAE_TILE_PIXELS` 保留為實驗覆寫；512px曾產生明顯格紋和偏色，不建議使用。
- 後續以相同 latent 實跑 ComfyUI `7c4390a6`／`VAEDecode`，比較原生編碼前 RGB8 與 Comfy PNG：107×768×768×3 共189,333,504個通道值，最大絕對差 **1/255**、PSNR **51.22dB**。這是全片 RGB8 層級的近似一致，不是 float tensor 或 bit-exact parity，也不排除兩套解碼器共同保留的瑕疵。
- 原生 raw 擷取後的診斷 mux 因臨時 wrapper 沒轉交音訊 fd 而失敗；完整 RGB 大小已核對，只用於上述像素比較，不採用其時間。完整原生出片及計時是另一個成功程序。

## 完整 ComfyUI 對照（2026-11-01）

硬體：同一台 M5 Max／128GiB，macOS27.0.1。原生功能版本 `c265111`，ComfyUI `7c4390a6f70cd8eb41014d6af52969397a752fc4`／0.43.0、PyTorch2.15.0、Python3.12.13。所有GPU工作串行持有 `/tmp/h3-gpu.lock`。

使用原驗收 manifest 的完整多行 prompt、同一圖片／音訊參考與首幀guide；768²／107幀／24fps，seed42、ER-SDE／sgm_uniform8步、shift12/3、全50層、F32 VAE256、BF16 DiT投影，未加LoRA。兩套亂數實作不同，相同seed不是相同噪聲，完整影片不作數值parity測試。

| 執行方式 | 完整出片時間 | 量測範圍 |
|---|---:|---|
| 原生，新程序 | 408.47s | 程序啟動、載入、生成、影音輸出；H3_PROFILE開啟 |
| Comfy，新程序，文字編碼CPU | 552.01s | 啟動5.02s，請求546.84s；DiT／VAE使用MPS |
| Comfy，新程序，明確文字編碼MPS | 832.86s | 啟動4.02s，請求828.10s；日誌確認MPS／BF16 |
| Comfy，常駐重跑seed43 | 380.97s | CPU文字編碼配置的第二個請求；沿用載入／conditioning快取，sampler及VAE確實重算 |

這是一組實測，不是多輪統計；process-cold不代表清空OS檔案快取。原生比本次較快的Comfy新程序少26.0%時間，但常駐Comfy與每job重啟的原生不是同條件warm比較。MPS文字配置反而較慢是觀察，尚未隔離其原因。

所有完整生成輸出均有107幀、768²、24fps、32kHz雙聲道音訊，FFmpeg完整解碼成功。抽看第0／26／53／80／106幀，兩套均保留單人、公園、全身及揮手；未從這組樣本認定任一套畫質普遍較佳，未做音訊主觀評分。

`/usr/bin/time -l` 的程序peak RSS／peak physical footprint：原生15.79／10.75GiB；Comfy CPU文字的三請求session95.83／108.09GiB；Comfy MPS文字單請求50.89／98.56GiB。這不是總統一記憶體或可直接相加的GPU配置量；session範圍不同，不推算每job記憶體倍率。

### 可重現安裝與產物

隔離安裝在 `ted:/Users/tedmacbookpro/projects/ComfyUI-h3-beta7`，沒有改動production或beta5 runtime：

- `h3-beta7-api.json`：完整API工作流；`h3-native-decode-api.json`：同latent解碼。
- `custom_nodes/h3_official.py`：官方14分片文字權重的名稱／前50層適配，以及校驗過容器／checksum的原生latent讀取；沒有改Comfy模型計算。
- `convert_h3_vaes.py`：AudioVAE172組weight_norm以PyTorch `torch._weight_norm(v,g,0)`實體化；影音VAE補入官方config的mean/std。音訊917keys、影片562keys與Comfy完整state鍵／shape相符，存讀張量一致，原始權重未改。
- 已還原較快的CPU文字適配器；明確MPS版本保存在 `h3_official-mps-tested.py`，不自動載入。沒有留下運行中的server。

從該目錄啟動（與原生同時使用時，外層仍需持有GPU鎖）：

```sh
.venv/bin/python main.py --listen 127.0.0.1 --port 8189 --disable-auto-launch \
  --highvram --bf16-text-enc --fp32-vae --use-pytorch-cross-attention
```

此機的SHARED策略下，以上旗標仍採CPU文字編碼；`--highvram`不能當作MPS文字駐留證明。API輸入的Autogrow欄位必須用 `ref_images.ref_image_0`／`ref_audios.ref_audio_0`。

本機產物在 `~/Movies/H3-Eros-Beta7-20261031/`：`原生-ComfyUI-並排對照-無音訊.mp4`、`native-cold.mp4`、`h3-beta7_00001_.mp4`、`ComfyUI-beta7-文字MPS.mp4`、`ComfyUI-同latent解碼.mp4`、計時JSON及完整log。遠端原始測量在 `/tmp/h3-comparison/`。

**決策建議：**固定Apple Silicon出片流程保留原生精簡後端，有本次量測的速度依據；新工作流／新模型優先用Comfy驗證，不為追上Comfy全部能力而擴張自建系統。這不是已證明原生畫質較好，也不是因既有投入而保留它。

## 原生加速查證（2026-11-01）

不改解析度、107幀、8步、50層、dense attention、參考圖／音訊或256px tile。原生已有權重駐留、融合MLP及M5 NAX路徑，不能把重新加入這些機制算成新優化。

### 實際形狀的DiT熱點

在隔離診斷執行檔中，對20,224 rows／56 heads／128 head_dim的完整50層forward逐操作同步計時。加入同步前後的latent容器逐位元組相同；同步版forward約34.56秒、未插同步約32.44秒，故下表只用來定位熱點，不拿診斷版當加速基準。

| 操作 | 50層累計 | 同步操作時間占比 |
|---|---:|---:|
| 完整attention | 14.814s | 42.96% |
| 融合MLP | 11.297s | 32.76% |
| QKV投影／norm／RoPE | 5.963s | 17.29% |
| attention輸出投影 | 2.201s | 6.38% |

`H3_PROFILE`的encode時間包含MPSGraph呼叫中的等待，root command buffer的GPU時間也不包含全部child buffer；原先encode約250秒不是CPU空轉250秒的證據。此單forward採initial latent／start_transition=7的controlled路徑，其額外dependency fingerprint與完整8步正常生成不同，整個probe程序耗時不能當完整出片時間。

### 既有後端設定篩選

每個設定單獨變動，均跑相同單forward及BF16 VAE；下表僅各一次，不將小幅差距當穩定加速。

| 設定 | DiT forward | 相對control的latent |
|---|---:|---|
| control | 32.687s | 基準 |
| `H3_ZERO_COPY_WEIGHTS=0` | 31.293s | 逐位元組相同 |
| `H3_DIT_COMMAND_BLOCKS=5` | 32.082s | 逐位元組相同 |
| `H3_REUSE_MPS_COMMAND=1` | 32.254s | 逐位元組相同 |
| `H3_NAX=mlp` | 33.630s | 不同 |

複製權重的DiT load由0.647秒增至3.747秒，單forward的DiT總時間反而由33.558秒增至36.124秒。這組結果不支持更改預設映射／提交／NAX策略。

### BF16 VAE是精度選項，不是無損新優化

同一原始latent、256px tile，既有BF16解碼實測約25.7秒，F32約88秒。編碼前全片RGB8（107×768×768×3）比較：PSNR60.414dB、平均絕對差0.05908/255、最大差4/255，約5.906%的色頻值不同；沒有差距大於4的色頻值。這不是float tensor parity或所有影片的畫質保證。

抽看第26／30／53幀人臉crop；BF16影片有107幀、768²、24fps及32kHz雙聲道AAC，FFmpeg完整解碼成功。F32仍為預設；選用既有`vae_precision: "bf16"`時，DiT精度、採樣步數與tile不變。比較影片已放到本機`~/Movies/H3-Eros-Beta7-20261031/原生-beta7-BF16VAE-256tile.mp4`。

### 完整attention分塊：不合入

重測既有slice-Q／完整K,V／concat設計，只改20,224-row的非causal BF16 DiT；每個query仍看見全部key，不是稀疏attention。隔離執行檔依序跑control、1024、4096、control，forward為32.170、32.024、31.698、32.114秒。4096的latent逐位元組相同；1024不同。

相對兩個control平均，4096僅少約1.38%，1024少約0.37%；每候選只有一次，未建立穩定收益，不新增正式開關，也不把這種差距宣稱成加速。這與[先前被否決的dense-query實驗](eros-beta5-quality-safe-dense-validation-2026-10-26.md)一致：公式完整並不保證更快或bit-exact。

### VAE雙tile batching：不擴張實作

先以真實F32 GPU矩陣運算量測可行性：每tile1797 rows，比較兩次1797-row投影與一次3594-row投影；相同總運算量，暖機後平衡ABBA，各4個樣本、每樣本8次。QKV（2048→6144）、FC1（2048→16384）、FC2（8192→2048）的batch2／逐tile時間比為0.9801、0.9904、0.9573；這組輸出最大差均為0。

這只是合成輸入的投影microbenchmark，不是完整VAE或影片加速證明。微小收益不支持增加batch-aware attention、RoPE與tile輸出重組的維護面，因此沒有修改正式VAE batching，也沒有把兩個tile混成一條attention序列。

### 完整8步配對驗收

重新從seed42開始生成，不用initial latent或跳步；兩路都使用未修改的`c265111`執行檔，只切換VAE精度。每路一個新程序，GPU鎖串行執行。

| 同輪實跑 | 完整出片 | DiT去噪 | VAE解碼 |
|---|---:|---:|---:|
| F32 VAE | 363.21s | 270.91s | 80.54s |
| BF16 VAE | 313.84s | 276.78s | 25.72s |

整體少49.37秒、約13.6%，不是整片快3倍。這是單次配對，不是多輪統計；同一F32程式的本輪363.21秒與先前408.47秒有差異，不能把跨輪波動算成程式優化，也不拿先前Comfy時間推算本輪倍率。

兩路latent容器及原始驗收latent的SHA-256皆為`f9fea08a54e0577ad6692d5af8e9831402eb7bcaabb277d857dc894e47e8c0ea`，確定採樣結果未因VAE精度改變。兩支完整影片均通過FFmpeg全片解碼：107幀、768²、24fps、32kHz雙聲道音訊。全片RGB8的精度差異使用上面的同latent對照，不以有損H.264輸出冒充float parity。

本機同一Movies目錄新增`complete-f32.mp4`、`complete-bf16.mp4`、`complete-results.json`與兩份完整profile log，另有`BF16VAE-RGB8比對.json`及`F32-BF16VAE-人臉對照.png`。

**本輪決策：**沒有合入無穩定收益的新效能程式碼或實驗開關；F32預設不變。需要這組已驗證的速度／精度取捨時，在既有manifest指定`params.vae_precision = "bf16"`。production與beta5 runtime未修改；完整比較與加速實驗的紀錄更新於本文件。
