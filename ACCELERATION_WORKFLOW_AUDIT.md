# Acceleration workflow audit

查閱日：2026-10-26。僅擷取文字／JSON與閱讀既有文件；未改引擎、編譯、測試、推論、部署或下載權重。使用者回報最新高解析度成果不滿意，視為品質問題的有效觀察，不以舊短測或尺寸標籤反駁。

## Actual pixelaigc graph, not its filename

已讀取 [實際 JSON][1]（Git blob `689877ecaad14095e4e93f81510ef4285da05e1a`）和 [README][2]（blob `19195fd88c9c0b1316086203fdc6f7e61cee0186`）。檔名含 `t2v`，但儲存狀態連接兩個 `LoadImage` 至首／尾幀；不是已知純文生影片 benchmark。

- 外層 node105：`minimax_h3_fl2va_pruned_int8_convrot`；`qwen3vl_32b_minimax_h3_nvfp4_awq`；video VAE **FP16**、audio VAE FP32；`minimax_h3_turbo_4step_ema_ckpt850`；**8 steps、6 seconds、16:9、0.4MP**。
- 子圖 node119 `MiniMaxH3TurboLoRA`：strength1、low_vram=false；node17 `KSamplerSelect=res_multistep`；node9 `BasicScheduler=simple`, denoise1；`BasicGuider`，不是額外CFG雙分支。
- node122 `ResolutionSelector` multiple32：有效 **864×480**。node107 公式 `max(5, round(a*24)) + (5-(max(5, round(a*24))%17))%17`：6秒→**158幀／24fps≈6.58秒**。不能把未接線的內層舊widget1344×768／73幀當有效設定。
- 結尾 `SamplerCustomAdvanced → VAEDecode / VAEDecodeAudio → CreateVideo(24fps,8bit) → SaveVideo`。
- **完全沒有 SageAttention patch、TeaCache、影像超解析、latent upscale／refinement、分段延伸／overlap 節點**。量化＋pruning＋Turbo確有設定，另外兩／三種技術不能從別人的描述補進這個圖。
- README明言未記錄GPU、峰值VRAM、後端revision、耗時。這張圖無法證實「7秒1080p、3分25秒」；該量測的硬體、精確graph、輸入／權重、精度、cold/warm及是否包含upscale仍未知。[1][2]

## Five-technology claim matrix

| 技術／主張 | 判定與來源 | Native Metal 可採取的動作 |
|---|---|---|
| 960×540生成後2×成1080p | **方法可行，指定graph不支持**：圖中沒有upscaler；540不是32倍數，圖的0.5MP示例是960×544。精確2×會成1920×1088，需明示crop／resize才是1920×1080。[1][3] | 低解析生成＋像素域超解析只能標示為另一交付路徑，不能冒充原生1080p／latent refinement；先比較細節與動作，不保證較好或較快。 |
| SageAttention快30–40% | **有CUDA H3個案，不是這張圖或Metal保證**：RTX3090作者9→6.57秒/iteration≈1.37× sampling，約27%少時間；resident每seed204→155秒約24%少wall。品質非bit-identical。[6] | 官方Sage針對NVIDIA Ampere/Ada/Hopper/Blackwell，依CUDA/Triton kernel；不能把Python patch移進Metal當現成功能。[4][5] |
| int8/pruned＋Turbo8–15步幾乎無損 | **部分支持、品質／步數泛化不支持**：指定圖確為pruned-int8＋Turbo，實際8步；並非所有Turbo都建議8–15步。pruning、量化、蒸餾是不同改變；pruned作者明言非bitwise，不能以小檔等比例推算速度。[1][7] | 原生已有Turbo，先保持匹配base／adapter／schedule；pruned BF16為獨立loader與LoRA座標相容性工作，不是rename檔案。int8_convrot格式也不等於本機既有int8 kernel／row-fc2。 |
| TeaCache1.5–2×且相容H3 | **查閱來源未建立**：TeaCache Comfy port支援表列FLUX、HiDream、Lumina、Hunyuan、LTX、CogVideoX、Wan2.1，沒有H3；1.5–2×不能移植為H3結果。[8] H3實測找到的是 **EasyCache**／Spectrum，作者記錄EasyCache高頻紋理損傷。[6] | 不加入預設；Metal重新實作也需AV／RES時間與cache reset身份正確，不能把跨步估計cache與精確weight／conditioning cache混稱無損。 |
| 22／39幀latent overlap完美連續 | **指定圖不支持；「完美」無證據**：22、39符合17k+5合法總幀數，不構成overlap算法證明。官方有continuation template，但它不是本圖的功能，更不是任何兩段可無縫的品質保證。[1][3] | 延續條件、去噪mask與拼接需明確設計；評估接縫動作、主體、音訊，不以frame grid推定保證。 |

## Official templates and KJ patch scope

直接擷取並解析官方 `video_minimax_h3_t2v.json`、`video_minimax_h3_i2v.json`、`video_minimax_h3_i2v_continuation.json`，固定tree revision **`67984637eefcd4825dcf5c134e39034785be1682`**。[3] 它們使用pruned-int8 base、nvfp4 encoder、INT8 ConvRot video VAE（與pixelaigc FP16 VAE不同）、RES/simple，含Turbo選擇／switch及20／6等primitive值；實際連線與switch優先於內層`BasicScheduler`儲存值4。不能只搜尋widget步數就宣稱官方固定4步。T2V／I2V採32對齊；I2V有**輸入圖片** `ImageScaleToTotalPixels`，不是生成後2×影片upscale。所檢模板沒有Sage／TeaCache／超解析節點。

KJNodes固定revision **`37ca7aff7ed14dde6ba3c0b25713e21c18b28dcb`**：`nodes/minimax_nodes.py`目前提供 `MiniMaxLowVRAMAttention`（head groups、提早釋放buffer）與 `MiniMaxChunkFeedForward`；前者經`optimized_attention`路由，不能僅憑節點名推定用了Sage。`nodes/model_optimization_nodes.py`的 `PathchSageAttentionKJ` 才選擇sage CUDA／Triton API。[4] 舊`MiniMaxH3MemoryEfficientSageAttentionPatch`有SM89灰雜訊及wheel `_qattn_sm89=None` issue；這些是回報，不能推定所有硬體都失敗，也不能忽視版本／sequence length。[10]

## Scaling and upscaling: precise interpretation

官方架構：VAE空間16×、再2×2 patch，故video token量近似 `latent_time × (W/32) × (H/32)`；另加文字、音訊及參考token。固定時長與步數，雙倍寬高約4×video tokens；dense attention主項約N²，可能約16×attention FLOPs，projection／MLP約4×。這是**多項式，不是指數成長**；實際wall含loading、encoder、decode、bandwidth與kernel，不等於任何單一比例。[9]

像素upscale在VAE decode之後，不重新用H3去噪；latent refinement需要額外VAE encode／latent處理及擴散步驟。官方H3-Regenerate-2K是以原內容與context進行生成式regeneration，**尚未開源**，不是普通ESRGAN或單純拉大畫面。[9] 因此不能把商用H3 2K與本地小圖2×當等價技術。另一作者RTX3090測480p＋RealESRGAN4×再縮720p總體反而慢13%；僅說明upscale成本不能忽略，並非否定使用者的另一個benchmark。[6]

## Metal-friendly acceleration candidates and existing coverage

| 候選 | 已讀證據／限制 | 判斷 |
|---|---|---|
| AdaLN預計算 | 官方33B中約13B為AdaLN，可預計算cache後不常駐。[9] 本repo既有DiT準備已含預計算。[11] | 原理Metal友善，**不是尚未採用的新13B整體加速**；新schedule／adapter必須有正確身份。 |
| Pruned rank8 BF16 | 維持AdaLN函數的低秩近似及table插值；pruned權重的LoRA input8與released2688座標不能混。作者H100 offload94vs205秒，明言全resident預期相同速度。[7] | 優先視為載入／容量候選；完整resident128GB不承諾2×，需正確映射AdaLN bias與LoRA，不能默默drop targets。 |
| Lazy Metal pipeline | 上游PR75 compile-on-first-use；本repo已port。既有本地paired wall136.023vs136.140秒，未證實通用startup收益。[11] | 保留現有精確cache，不重列新功能或外推作者14%。 |
| Residency／conditioning cache | 作者24GB GPU重複seed大幅節省reload與encoder；本repo已有resident及身份約束。[6][11] | 依cold/warm及prompt變更計收益，不把repeat-seed倍率當首片倍率。 |
| BF16 video VAE | 本repo已提供選用BF16且保留F32預設，既有39f768p paired完整wall約少18.6%，不代表所有畫布／長度。[11] | 目前可用、無新權重；非LightVAE、非無損，audio／reference encoder不跟著改。 |
| LightVAE | ModelTC H3為**block-pruned trained encoder／decoder**，Comfy整合沿用native normalization／stitching／offload，不需自訂Triton。FL2V建議只換decoder，音訊VAE不替換；預設1344×768124f20步，不是Turbo benchmark。可選INT8版CUDA13建議不能套成Metal倍率。[12] | 架構可port Metal，但本repo未實作；需retained-block mapping／weights與quality驗收，不能只把原VAE改精度後稱LightVAE。 |

**品質優先決策：**保留使用者指出的品質落差，先區分prompt/context、匹配模型／sampler、原生畫布與post-upscale的成品差異；選用現有resident／BF16需披露數值差異。不要一次疊Sage替代kernel、int8新格式、TeaCache與upscale後宣稱「幾乎無損」。本研究不啟動以上候選的實作或benchmark。

## Sources

[1]: https://raw.githubusercontent.com/pixelaigc/ComfyUI-MiniMax-H3-Workflow/main/video_minimax_h3_t2v_0909.json
[2]: https://raw.githubusercontent.com/pixelaigc/ComfyUI-MiniMax-H3-Workflow/main/README.md
[3]: https://github.com/Comfy-Org/workflow_templates/tree/67984637eefcd4825dcf5c134e39034785be1682/templates
[4]: https://github.com/kijai/ComfyUI-KJNodes/tree/37ca7aff7ed14dde6ba3c0b25713e21c18b28dcb/nodes
[5]: https://github.com/thu-ml/SageAttention/blob/main/README.md
[6]: https://github.com/pepikir/minimax-h3-speedup/blob/main/README.md
[7]: https://huggingface.co/multimodalart/MiniMax-H3-Pruned/blob/main/README.md
[8]: https://github.com/welltop-cn/ComfyUI-TeaCache/blob/91dff8e31684ca70a5fda309611484402d8fa192/README.md
[9]: https://github.com/MiniMax-AI/MiniMax-H3/blob/main/README.md
[10]: https://github.com/kijai/ComfyUI-KJNodes/issues/712
[11]: UPSTREAM_RESEARCH.md
[12]: https://github.com/ModelTC/LightVAE/blob/main/comfyui/README.md

Additional exact sources: [Sage wheel incompatibility issue736](https://github.com/kijai/ComfyUI-KJNodes/issues/736); [current local usage／status](README.md); [local resident and BF16 evidence](COMFY_CACHE_RESEARCH.md). Author-reported source measurements above were not reproduced in this audit. GitHub/HF `main` sources are mutable; recordedblob/tree identities delimit the fetched graph and template observations.
