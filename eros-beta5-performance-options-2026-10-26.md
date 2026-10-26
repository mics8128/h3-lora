# Eros Max beta5 效能設定與加速選項

範圍：ted／M5 Max128GiB，完整BF16 beta5 TURBO-hybrid來源，原生Metal Ref2VA。影片程序記憶體目標約40GiB，可小幅超過，保留語言模型及系統餘裕。使用者已暫停影片生成，本整理未啟動任何新推論；以下是既有結果與來源研究。

## 結論

使用者已選定快速4步：目前M5原生INT8 projection kernels＋BF16 videoVAE、4步ER-SDE/beta57、50層、reuse/core1、原生768×1344/24fps。這是設定採用，不是畫質、音色或嘴型核准；影片生成仍暫停。若要同解析度大幅快於目前，下一個研究主體是長序列DiT／attention，不是單純增加RAM、常駐或換VAE。

使用者後續要求的三個Astra獨立核心評估，見[整合決策](eros-beta5-astra-optimization-plan-2026-10-26.md)；其中明確區分完整dense精確語義候選與sparse/W4A8近似研究線。僅完成設計，沒有修改原生核心或恢復影片生成。

## 實測界線

一般公園單一成年人物圖＋音訊、141幀/5.875秒：6步733.205秒，去噪653.864秒、videoVAE64.663秒；4步489.602秒，去噪413.015秒、videoVAE53.973秒。每0.25秒抽樣phys_footprint峰值26.358/26.309GiB，非全系統或瞬時記憶體上限。各組一次、非獨占、filesystem cache未清；不是雙人應用端到端或LLM並行驗收。完整A/V解碼通過，4步尚未視覺核准。[應用實測記錄](https://github.com/mics8128/video-maker/blob/main/docs/VERIFICATION.md#2026-10-26-beta5效能參數對照)

去噪占6步樣本約89.2%。即使把videoVAE成本完全歸零，該樣本端到端理論上限也只約1.097倍；常駐省載入與編碼，同樣不能消除這段去噪。[原始phase log](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/outputs/perf-20261026-memory40-long/fast-six/stderr.log)

## 可選路徑

原生DiT attention呼叫MPSGraph `scaledDotProductAttentionWithQueryTensor`；repo未揭示framework內部是否materialize N×N scratch，不能憑API名稱斷言已用FlashAttention或一定建立完整score矩陣。文件中2048/3072 rows的門檻是部分BF16 projection快路徑，不是attention backend切換。需以真正長序列的GPU profiler拆解時間後才選kernel改法。[SDPA實作](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_gpu.m)、[DiT呼叫](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_dit.c)

| 選項 | 狀態 | 決策／代價 |
|---|---|---|
| M5 INT8投影＋BF16影片VAE | 已選用、原生實跑 | 數值不同；不是作者ConvRot checkpoint，不宣稱無損 |
| ER-SDE/beta57 4步 | 使用者選定、原生已有實跑 | 作者列4–6步；動作/聲音/一致性尚未品質核准，未重跑影片 |
| ER-SDE/beta 4步 | 引擎與作者均有此選項，未測本案 | 不是只換名字；sigma schedule改變，不保證優於beta57 |
| RES/simple6–8步 | 作者偏向動作品質，未測本案 | 品質候選，不先稱為加速 |
| LCM/simple6–8或LCM/beta4–6 | 作者列出，未測本案 | 音訊候選；不是1–2步保證 |
| 語言模型可常駐、重推理錯開 | 操作策略、未實作全域排程 | 保有RAM不代表沒有GPU算力/頻寬競爭；不自動停止其他服務 |
| native常駐context | 互動CLI已有，video-maker每job仍新process | 精確重用權重/conditioning，可提高連續試片效率；接入應用需開發並量測常駐高峰 |
| 較大VAE tile | 原生環境開關已有，應用未接入、未測 | 自動320px；可測384/512，可能少重疊也可能因attention變慢，改變邊界上下文，無保證 |
| 加深Qwen prefetch | 原生已有，應用未接入、未測 | 已M5自動depth3；最多6，增加記憶體，只影響編碼階段 |
| 降成960×544/544×960 | 原生進階尺寸可用，未測本案 | 直接減token量；必須標為低解析度草稿，不靠放大冒充原生768p |
| row-FC2 | 原生選用數值近似，未測本案 | 歷史其他模型收益小，不當主要方案 |
| LightVAE | 尚需原生loader/retained-block實作與權重驗證 | 僅加速decoder；不優先於89%去噪瓶頸 |
| Metal長序列attention優化 | 需profile/kernel工程 | 優先查attention/projection/同步各占多少，不能由phase time直接斷定attention單獨占比 |

## 不作目前預設

不疊額外Turbo（hybrid已融合）；不把beta6的10/3移作beta5作者建議；不套1–2步LCM；參考生成不啟用Spectrum/近似denoising reuse；不裁層/縮token後宣稱等價。NVIDIA SageAttention/CUDA與ConvRot檔案皆不能當現成Metal開關。[作者README](https://huggingface.co/TenStrip/10Eros-Max/blob/main/README.md)

## 資料來源

- [應用現行profile](https://github.com/mics8128/video-maker/blob/main/profiles/h3.json)、[應用單次process呼叫](https://github.com/mics8128/video-maker/blob/main/src/video_maker/h3_remote.py)、[render identity與提交](https://github.com/mics8128/video-maker/blob/main/src/video_maker/h3.py)。
- [作者beta5/beta6說明](https://huggingface.co/TenStrip/10Eros-Max/blob/main/README.md)：4–6步與reference cache警告；動態main來源，不當固定benchmark。
- [已安裝runtime reference](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/RUNTIME_REFERENCE.md)：互動context、SSD、precision、tile、M5 kernels及queue每job新process的契約。
- [已安裝beta5 setup](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/EROS_BETA5_SETUP.md)：來源、F32final契約及歷史baseline；此文件的BF16 preset並未被專案profile覆寫回寫。
- [LightVAE作者](https://github.com/ModelTC/LightVAE)、[本地既有評估](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/COMFY_CACHE_RESEARCH.md)：H200數字不能移作M5端到端倍率。
