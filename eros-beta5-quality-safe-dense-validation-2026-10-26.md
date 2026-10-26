# beta5完整dense query分塊：數值與速度實驗

## 決策

**本輪不採用、不新增正式環境旗標。** 依Luna發現的MIT PR77原始slice/concat機制，在獨立APFSclone實作完整K/V可見的query分塊，測過後未建立穩定加速。不是把稀疏attention當dense；也沒有更改正式runtime、模型或4步profile。實驗來源已封存，不把沒有收益的研究旗標永久留下。

## 實作與測試範圍

隔離目錄：`/Users/tedmacbookpro/projects/h3-perf-dense-query-20261026`。改動為h3_gpu.m內部graph adapter：head-major canonical Q以MPSGraph slice沿query axis分塊，K/V仍全部可見，concat回同一個完整output；causal不切；cachekey包含queryrows；disabled-by-default；graphparts上限128只限制nodecount，**不是frameworkscratch硬上限**。沒有offset假裝跨head contiguous的錯誤切法。

新weight-free probe透過既有public SDPA interface，先poison全output、檢查全值finite、按完整output計與baseline差異、A→B→A cache隔離；N7/129/257涵完整獨立double-softmax oracle與tailquery、row-major及head-major output。N8193及示例N45001使用56heads、D128；CPUoracle僅first/middle/last rows/heads，非全tensordoubleoracle。新5圖2音訊工作的精確N未量；N45001僅代表正式量級示例。直接head-major QKV producer串接與完整影音品質未在本輪驗收。

預先kernel門檻：sample/fulloracle maxabs≤0.003、relativeRMS≤0.005；split-vs-baseline relativeRMS≤0.005。這是實驗數值gate，不等同人眼/音訊差異門檻；沒有因結果修改容忍值。全部技術gate通過，仍不能宣稱全工作品質已核准。

## 平衡量測

每路徑各一次warmup，之後ABBA/BAAB八次、各4samples；使用同一QKV與context，poison/readback/CPUoracle不計turnaround。非獨占、只有一次balancedcycle，不能把小差距解讀成穩定收益。

| N／queryrows | baseline mean | split mean | candidate/baseline | output差異 |
|---|---:|---:|---:|---|
| 8193／1024 | 0.041699s | 0.048208s | 1.156087 | fulloutput相對RMS0.000905630 |
| 45001／1024 | 1.500373s | 1.488571s | 0.992134 | fulloutput相對RMS0.000792734 |
| 45001／2048 | 1.483163s | 1.634921s | 1.102321 | 此fixture全bytes相同 |
| 45001／4096 | 1.586599s | 1.583225s | 0.997873 | 此fixture全bytes相同 |

1024在長例只差約0.79%、4096只差約0.21%，同輪有明顯clock/負載漂移，沒有足夠證據建立有效加速。2048慢約10.23%；8193小例慢約15.61%。不採用任一queryrows作最佳通用值，不將這些SDPA-only數字冒充整支影片耗時。

N45001的CPU sampled-reference relativeRMS：原路徑約0.001676759；query1024約0.001681350；2048/4096與原始fixture輸出相同。1024的68,326,060個BF16元素與baseline不同，雖相對RMS很小，仍不是bitexact或影音品質核准。

## 記憶體與停止條件

長例launcher每0.05秒抽樣Darwinphys_footprint，三個長例均峰值約6.776GiB；36GiB是本次無權重probe的保護停止線，**不是改用戶40GiB軟預算或新增正式runtime硬限額**；未觸發停止。未載完整H3權重、encoder、VAE或LLM，因此不能當生成程序總峰值。

沒有做去噪中分段影片、刪frame、KV截短、sparse、crossstepreuse、W4A8；無任何影片或模型推論工作啟動。這是執行真實SDPA kernel的非影片smoke/微量測。

## 保存

遠端`outputs/dense-query-rejected-20261026.tar.gz`保留candidate h3_gpu.m、Makefile、THIRD_PARTY_NOTICES.md、probe與guardedlauncher；包含PR77固定commit attribution及既有MIT LICENSE引用。本次session工具輸出保存small/8193/45001/2048/4096所有結果，原始測量不是新媒體。正式native部署未修改；數值微差但無收益的候選不應只因「能編譯」而合入。

來源：[PR77固定patch](https://github.com/flysheep-ai/h3.c/commit/0a95ea10c0cab51d16ef094ee88dba898e23524d)、[Luna分支整理](eros-beta5-luna-forks-platforms-2026-10-26.md)、[Astra streaming分析](eros-beta5-astra-streaming-2026-10-26.md)。
