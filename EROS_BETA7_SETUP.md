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
