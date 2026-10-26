# beta5品質保守核心優化：FC2實作與否決記錄

## 決策

**不採用FC2 K1024 specialization。** 已在隔離runtime實作並完成逐位數值驗證，但平衡順序的長列數測量沒有穩定收益；不以第一次固定順序約2%的差距宣稱加速。原生正式runtime、模型與專案profile未因此變更；隔離active source亦已還原，候選保留為可重現實驗檔而非永久旗標。

## 改動與正確性

隔離目錄：`/Users/tedmacbookpro/projects/h3-perf-quality-safe-20261026`，從正式beta5 runtime作APFSclone。候選只把每個1024-channel scale group的兩個K512 INT32乘法合為一個K1024；原INT8權重、F32 scale/group累加次序與BF16輸出不變，不新增模型權重副本。此前實際Astra source review允許隔離數值實驗，但指出baseline同library、未poison輸出及unitweightscale測試不足。

依審查已補：每次呼叫前BF16NaN poison、原始未改shader獨立context全輸出bytes對照、逐channel power-of-two非unitweightscale、zero-input row/group、default與slow groupedquantizer、獨立CPU量化及integer-dot oracle。每個暖身與量測迭代都比對完整輸出，檢查尾端列；readback/poison不計kernelturnaround。

rows128、129、257及示例45,001，productionwidth5376→14336→5376均與原始source完整bytes一致，CPUoracle通過。45,001是示例長列數，**不是實際雙角色工作已觀測seq_len**。

## 平衡量測

同一context、暖身後ABBA/BAAB三輪，每路徑每quantizer12個量測。主機非獨占，數字是whole-public-MLP-seam synchronous turnaround，不是FC2-only或整支影片時間。

| 45,001 rows | K512 mean | K1024 mean | candidate/baseline | 判斷 |
|---|---:|---:|---:|---|
| production default grouped quantizer | 0.203356s | 0.203962s | 1.002979 | 約0.30%較慢，不採用 |
| diagnostic slower grouped quantizer | 0.203501s | 0.202936s | 0.997226 | 約0.28%差距不足以建立收益 |

此測量process `time -l` peak memory footprint為9,422,040,200bytes（約8.78GiB），maxRSS16,168,615,936bytes為另一口徑，不相加。不是包含完整H3模型或LLM的工作峰值。

## 執行過的其他檢查

隔離build無觀察到警告；sampler、beta5curve/rank8/Torchfinalprecision、Multi-SIMD GQA/encodernorm、VAEprecision檢查通過；host tests回報5,293checks。這些非影片檢查不等於完整影音品質核准。

## 保存與回復

隔離runtime的`outputs/quality-safe-fc2-rejected/`保留`experiment.patch`、candidate source/test/Makefile、outcome.json、restored-source-sha256.txt與已測試executable/object。local mirror已移至h3專案`outputs/quality-safe-20261026/local-mirrors/native-quality-fc2-rejected/`，不留在video-maker。還原後`h3_gpu.m`、`h3_shaders.metal`、Makefile SHA256與唯讀baseline一致；未重編還原來源，故不宣稱隔離binary已是baseline。正式runtime從未被替換。

原始執行輸出保留於本次session artifact76；archived outcome.json記父代理實測值。沒有重新生成影片、停止其他服務、改採樣步數、裁素材或縮圖。
