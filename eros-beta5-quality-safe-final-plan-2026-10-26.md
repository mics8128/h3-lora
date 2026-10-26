# Eros beta5 品質優先加速：本輪結論與串流路線

本輪h3核心研究、候選程式與微量測歸h3專案；video-maker只維護應用profile與操作／驗證說明。十二份研究文件與本頁同層，原始候選、公開fork資料及短片量測鏡本位於h3專案忽略目錄`outputs/quality-safe-20261026/local-mirrors/`；`outputs/quality-safe-20261026/relocation-sha256.json`記錄搬移前後相同的37檔SHA256。沒有把被否決候選合入正式核心，也沒有提交權重、影片或大型原始JSON。

## 現在採用什麼

video-maker應用維持使用者已選的 **4步ER-SDE／beta57、M5原生INT8投影、BF16影片VAE、F32音訊VAE、50層、reuse/core1、原生畫布與完整素材**，不是改寫h3全域預設或安裝preset。記憶體目標約40GiB、可小幅超過；LLM可常駐，但重GPU推理仍會競爭算力與頻寬。最佳已知自動值為Qwen8workers/depth3、command split30/50、自動VAE320、M5zero-copyauto，沒有新增激進覆寫。[設定與品質門檻](eros-beta5-best-settings-quality-safe-2026-10-26.md)

**本輪沒有把任何新核心patch合入正式runtime。**不是未測便放棄：兩項候選均已在隔離APFSclone完成實作與非影片量測，因無穩定收益而否決，封存後還原active來源。沒有新影片、沒有停止服務或改動模型。

## 已實際驗證與否決

| 候選 | 實測結果 | 品質／決策 |
|---|---|---|
| FC2 K1024，45,001示例rows，production-width MLP | defaultquant平衡12samples/path，0.203356→0.203962s，約慢0.30% | 與原始未改shader全bytes相同、獨立CPUoracle通過；無速度收益，不採用 |
| Full-KV query1024，N45001、H56、D128 | balanced4samples/path，ratio0.992134 | 全output相對RMS0.000792734；差距不足以建立速度收益，未感知驗收，不採用 |
| Full-KV query2048 | ratio1.102321 | 此fixturebytes相同但約慢10.23%，不採用 |
| Full-KV query4096 | ratio0.997873 | 此fixturebytes相同但無穩定有效收益，不採用 |

這些是**kernel/wholeMLP微量測，不是影片端到端耗時**；N45001是正式量級示例，不是雙角色新工作已量到的seq_len。沒有exclusiveGPU保證。dense長例每0.05秒抽樣footprint約6.776GiB，未載完整模型；不能當成H3＋LLM工作峰值。[FC2完整證據](eros-beta5-quality-safe-fc2-validation-2026-10-26.md)、[dense完整證據](eros-beta5-quality-safe-dense-validation-2026-10-26.md)

隔離sampler、beta5curve/rank8/Torchfinalprecision、Multi-SIMD reduction、VAEprecision及5,293個hostchecks通過。FC2進一步覆蓋poisonedoutput、原始shader對照、非unitweightscale、zero-group、default/slowquantizer及ABBA/BAAB。這些數值證據不能代替人物、動作、中文台詞、音色、嘴型與AV品質驗收。

## Luna蒐集與Astra細查做到哪裡

Luna實際諮詢兩次：先篩選，之後在認證API可用後補齊全部回傳fork的branch/commitmetadata。Astra實際portability諮詢兩次：第一輪逐項A0–A6；第二輪只深入新增固定來源。另有獨立Astra串流分析。

- forkAPI回傳243個；全部243個branchendpoint均嘗試，240成功、3個明確404。
- 345branchrefs、134distinctheads；全部133個非baselineheads完成compare與commitpagination，無compare/pagination錯誤。
- 1,338個distinctcommits；一個head觸及GitHub300file清單上限。
- 47筆upstreamPR；另外看ComfyUI、LightX2V、Diffusers、PyTorchSDPA、FlashAttention、AppleTensorOps的原始文件／程式。

這是完整可達**metadata discovery**與挑選相關固定程式的深入閱讀，不是243個fork每行程式、私有／不可達分支、每個授權或benchmark的全面審核。[Luna清單與誤差／缺口](eros-beta5-luna-forks-platforms-2026-10-26.md)、[Astra兩輪固定來源細查](eros-beta5-astra-luna-portability-2026-10-26.md)

### 大平台的可借鏡與不可直接套用

ComfyUI主要有相容的packedAV/多reference語義、optimized-attention seam及模型／conditioning常駐。LightX2V與Diffusers可組合fusedattention、量化、offload與SP/TP多GPU；它們的CUDA/Sage/FP8配置不是M5Metal現成加速，亦不是品質中性。FlashAttention提供完整dense的IO-aware演算法，不代表現有MPSGraph沒有tiling；AppleSDK27提供更強cooperativetensor方向，但新kernel仍需本機compile/numeric/perf/qualitygates。不可將「有Flash/NAX名稱」當成已勝出：Luna找到的多條公開dense實作已有輸給MPSGraph的反證。

## 去噪串流應分開處理

1. **SSD權重串流：不開。**現有實作會停用目前INT8projection路徑；主要換記憶體，不是速度優先。
2. **完整dense attention工作集串流：數學可行，速度未證。**Querytile讀完整KV；KVtile要以全域online max/sum/weighted accumulator合併，不能各自softmax再相加。本輪PR77單graphslice/concat未勝出，而且不等於boundedworkspace或host雙槽。
3. **GPU命令與activation生命週期：可研究，先補正確歸因。**MPSGraph有child/selfcommit，root timestamp不等全部GPU時間；`4aed086c`值得借鑑診斷設計，但root換手不能自動視為完整childcoverage，也不因此改queue深度。
4. **最終影格串流：比較可取，但在完整去噪之後。**H3為noncausaljointattention，不能把「一段先去噪完就丟掉」說成同品質串流。完整denoise後，可以保留原spatial/temporalblend，逐批VAEdecode／轉RGB／送FFmpeg，減少峰值與更早交付已確定影格。

GPU-stateER-SDE仍後置，需保BF16velocityrounding、packedAV雙clock、history/RNG/checkpoint/init順序；不能把消除hostwait當成消掉實際GPU運算。[Astra串流完整研究](eros-beta5-astra-streaming-2026-10-26.md)

## 新來源最值得保留的四個小改動方向

Astra讀完Luna新增來源後，建議從[Bambushu b0966252](https://github.com/Bambushu/h3.c/commit/b0966252a375fb464f4046276f63af53b8ccc626)拆開，而不是整包移植：

1. **MPS診斷**：rootreplacement、encode、command-create/backpressure分開記帳，不更改queue行為。[4aed086c](https://github.com/Bambushu/h3.c/commit/4aed086cd8e116783fad7d32936e8b872599778a)
2. **VAE等算術epilogue**：先RMSNorm→BF16，再bias＋SwiGLU／residual；逐stageBF16/F32bytes對照，不改原reduction與rounding，不碰DiT F32final。
3. **VAE雙槽**：只input/projectedoutput雙槽，共用一套weights/scratch，在單orderedqueue以completion退休後重用；不是兩套VAE同時算。GPU下一tile可與CPUunpack/stitch重疊，但fence/error/cancelownership必須驗證。
4. **最終frame sink與FFmpeg feeder**：保持raster順序、先top再leftblend、22/17/5frame組裝、最後五幀、PCM samplecount與AVduration。不能少尾幀或用到達walltime重寫timestamps。

Astra找到具體不可整包搬入的差異：來源unpack採`fminf/fmaxf`，本機用conditionalclamp，對NaN/signedzero不等價；若移植assembly必保留本機語義，不能把NaN掩成合法像素。來源「bit-identical」是對其來源樹的聲明，不是本機baseline認證。新epilogues、雙槽、FFmpegtail均尚未實作/測量/部署。

上述偏向decode尾段與省記憶體，**不是把去噪本身變成逐幀、也不承諾大倍數提速**。先前四步樣本84%時間在denoise，VAE優化僅影響其本身與pipelineoverlap；不把作者decoder倍率移作整片倍率。

## 目前不採用

Sparse/local/VSA、W4A8、Sage/FP8attention、crossstepKV/residualreuse、裁層／縮圖／少素材，均不因「極度輕微可接受」而重新開放。真正weightprepacking仍未實作量測；FC2K1024的否決不能推論prepack也無收益。原生dense新kernel亦沒有本機勝出證據，不以kernel名稱或外部CUDA數字替代。

結案狀態：四步快速設定保留；新增core候選不採用；Luna→Astra詳細研究完成；串流與最佳已知設定文件完成；影片暫停持續。未感知驗收的數值候選不能升正式預設。
