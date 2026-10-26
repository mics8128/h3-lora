# Eros beta5 四步最佳已知設定與極輕微差異門檻

## 決策

目前保持已選四步快速設定；不新增未驗證的全域環境覆寫。這是「最佳已知且有本機證據的起點」，不是所有輸入與並行負載下已證明的全域最優。使用者允許極度輕微差異，但未授權中等品質取捨；sparse、W4A8、跨stepKV/residualreuse、少圖、裁層、縮圖均排除本輪採用。

## 可直接使用的設定

| 項目 | 選定值 | 為何不再加碼 |
|---|---|---|
| 模型 | 現有完整BF16來源ErosMax beta5 TURBO-hybrid Ref2VA | 不換權重、不另疊Turbo；source BF16不表示execution全BF16 |
| sampler/scheduler | ER-SDE／beta57 | beta5作者列4–6步；本機已有4步技術實跑，不把beta6 shift當建議 |
| 步數 | 4 | 使用者明確選定；不是1–2步研究 |
| layers/reuse/core reuse | 50／1／1 | 保留完整denoising計算，不依cache猜跨步結果 |
| 畫布 | 原生768×1344直／1344×768橫，24fps | 不靠降規格製造加速數字 |
| projection | M5原生INT8，`bf16_projections=false` | 已有本機精度/速度對照；不是作者ConvRot權重 |
| video/audio VAE | BF16影片／F32音訊 | 維持既有已採用選項；不加INT8VAE或LightVAE |
| shift | 12／3 | 既有nativefallback，不宣稱beta5作者最優 |
| 參考圖與音訊 | 保留全部選定、原生match | 不以刪素材或任意縮圖換速度；不縮成無聲路徑 |
| 記憶體 | 影片約40GiB軟目標、可小幅超過 | 不制定新硬限額，也不把剩餘全當可用GPU工作空間 |

這些是video-maker應用選用值，不是對h3全域預設或已安裝preset的改寫。現行profile與離線準備證據：[video-maker profiles/h3.json](https://github.com/mics8128/video-maker/blob/main/profiles/h3.json)、[應用驗證記錄](https://github.com/mics8128/video-maker/blob/main/docs/VERIFICATION.md)。

## 維持原生自動值，而非強迫最大值

- **Qwen預取**：M5預設depth3、8個I/O workers；depth上限6不等於最快。已量樣本主要成本是denoise，增加預取只可能影響encoder，會增加live buffers。除非phase log顯示讀取等待主導，先不改。[text_prefetch_threads/text_prefetch_depth](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_text_encoder.c:124-142)
- **Command split**：50層M5預設30層處commit，非每層wait；不把`H3_DIT_COMMAND_BLOCKS=0`當作通用加速。5/10/30等窗口需同outputbytes與完整GPUtimeline對照後才選。[command_block_interval](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_dit.c:213-224)
- **Zero-copy**：M5原生權重映射是既有策略，不強迫關閉；INT8量化及當前權重持有方式另計，不能拿37GiB文件大小當程序總峰值。[native residency reference](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/RUNTIME_REFERENCE.md:741-763)
- **VAE tile**：768×1344自動選320px、3×5tiles；維持auto。把tile加到384/512會改變decoder上下文與workspace，可能改畫面或變慢，不能當作純記憶體無損優化。[configured_tile_pixels](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/h3_video_vae.c:770-800)
- **NAX**：不要強迫`H3_NAX=1`或`mlp`。原生已有適用的M5快路徑與fallback；既有文件記錄某些完整forward反而退步，BF16NAX MLP不是目前INT8預設的免費升級。[Metal4 paths](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/RUNTIME_REFERENCE.md:765-793)
- **SSD streaming**：關閉，當前INT8resident流程沒有啟用。這是低記憶體BF16權重串流，不是提速開關。[SSD streaming](ssh://ted/Users/tedmacbookpro/projects/h3-lora-eros-beta5/RUNTIME_REFERENCE.md:233-260)

## LLM共存

分開管理memory admission與compute admission。LLM可以載入，但影片去噪和LLM大量prefill/decode同時執行會搶GPU算力、頻寬與功耗；40GiB影片目標不保證兩邊同速。本次不停止或卸載既有服務、不新增虛假的全域排程器。

真正預算應包含LLM權重＋KV＋scratch、影片phasepeak/frameworkscratch、macOS/other服務與餘裕。先量實際memory_pressure/swap增量與processphysfootprint，不能只看RSS、不能把filemapped與Metal計帳簡單相加。主機本輪只讀查詢為macOS27.0.1 build26A434、128GiB；歷史macOS26.x kernel倍率不直接移作現況保證。

## 極度輕微差異如何判斷

1. 純排程、權重重排、INT8整數乘法group合併：以現行已選nativeINT8為oracle，要求相同INT8/scales與BF16輸出bytes，不放寬成僅cosine近似。
2. 完整denseattention的online-softmax／querysplit：即使所有attention邊保留，浮點累加順序也可能不同。先量獨立高精度oracle、MPSGraph基線、按head/modality的誤差與尾端；不能僅平均PSNR宣稱品質不變。
3. 數值誤差小仍要人物/說話者、動作、台詞、音色與AV同步媒體驗收。使用者影片暫停仍有效，本輪未做該驗收的路徑不升正式預設。
4. 不先捏造通用數值門檻代表人眼無差；每seam預先訂定門檻，失敗修正或拒絕，不能看完結果才放寬。極輕微可接受並不授權sparse/W4/cache。

本輪新增核心候選在隔離runtime驗證，正式runtime與模型先保持不動；成果與失敗實驗另記錄，不隱藏未知或回退成假成功。
