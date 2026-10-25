# Ref2VA 多模態參考與 LoRA 相容性研究

研究日期：2026-10-24。僅查閱第一手文件與程式碼；未建置、執行測試或推論。以下「程式路徑支援」不代表已驗證生成品質，也不等同每種 Turbo LoRA 都曾針對此組合訓練。

## 結論

- **純音訊輸入、生成影音的 Ref2VA，存在完整的 ComfyUI 上游程式路徑，不必偽造圖片。** 相反地，ModelTC 的 Diffusers resize wrapper 明確禁止只有音訊；h3.c 文件也要求音訊伴隨圖片或影片。不能把這三個實作混稱為相同限制。[1][2][3][4]
- 多人物／商品圖片、動作或運鏡影片、影片搭配原音／替換音訊都屬於 Ref2VA；需載入 Ref2VA checkpoint，而不是把 FL2VA 的 first/last-frame 分支當作一般參考。[5][6]
- LoRA 的「Diffusers、Comfy、native」不只是副檔名不同：module 名稱、QKV 融合方式、factor suffix、alpha/rank 縮放均須核對。[7][8]
- Ref2VA Turbo8 不應套用 FL2VA 的 video shift=6 建議。現行本地 v1.0 preset 的 12/3 是採用 Comfy 預設；不是已取得該 Ref2VA 版本訓練 shift 的證明。[9]

## 1. 純音訊 Ref2VA：實際程式碼證據

固定 ComfyUI revision：`7cebca52227c63b4180ceed448cbd7ad22903b6d`。

### 輸入、編碼與 conditioning

[`MiniMaxH3ReferenceToVideo.execute`][1] 的 `ref_images`、`ref_videos`、`ref_audios` 都可為 None；沒有要求至少一項視覺參考。前兩個迴圈可以完全不執行，獨立音訊迴圈仍執行：

```python
for audio in (ref_audios or {}).values():
    if audio is None:
        continue
    ref_items.append({"type": "audio"})
    if audio_vae is not None:
        audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, audio)
        ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t,
                           "audio_latent": audio_latent})
```

後續呼叫 `clip.tokenize(prompt, minimax_ref_items=ref_items)` 並將非空 `ref_blocks` 放入 `minimax_refs`。`_encode_ref_audio` 會轉到 audio VAE sample rate（預設 32000），僅取 waveform 的第一個 batch，輸出 `[1,32,2,T]`。**必須連接 audio_vae，否則只有文字標籤，沒有實際聲學 latent。** 節點 tooltip 所謂「audio only conditions the text encoder」不表示 Qwen 聽到波形；實際 tokenizer 只看標籤。[1][2]

### 沒有圖片時的文字、token tags 與 RoPE

[`MiniMaxH3Tokenizer.tokenize_with_weights`][2]：

```python
elif kind == "audio":
    add_text("<Audio %d>: " % counters["audio"])
# ...
add_text(text)
```

- 檔頭明文說 **NOT chat-templated**；標籤和 prompt 是 raw text，不套 chat template。
- 音訊沒有特殊 audio token，也不把波形餵入 Qwen；`<Audio 1>: ` 是一般文字。該檔列出的額外特殊 token 是 dialogue／lyrics／caption 等，沒有 Audio reference token。
- `token_tags_from_embeds_info` 先建立全 1 的 tags，只有 vision/image spans 變 0。因此純音訊 presentation 的文字 tags 全 1。
- [`PackedLayout.__init__`][3] 的 text position 為 `g=zeros(text_len,3)`、`g[:,0]=arange(text_len)`，即 `(t,h,w)=(i,0,0)`。這是 **DiT packed text** 的位置；Qwen 純文字本身使用其既有純文字編碼路徑，不應混同兩層 RoPE。
- 同一 PackedLayout 先從**目標 canvas** `_frame_grid(latent_h, latent_w)` 取得 `target_audio_w`；`kind == "audio"` 用 `_audio_grid(cursor, rt, *target_audio_w)`，完全不依賴 reference image 大小。音訊區段 `rt*2` 列（左右聲道 channel-major），cursor 前進 `rt`；然後接目標 audio、目標 video。
- `img_pos` 最終仍有目標 video rows，故沒有 reference image 不會令圖片位置清單完全為空。[3]

這些是「只有音訊輸入」而不是「圖片加音訊」的明確 code-path 證據；本次未做聲畫同步、音色保持或內容遵從的推論驗證。

### 相反證據：不能宣稱所有上游入口都開放

ModelTC 固定 revision `02e26d591f7a04d5d1a074c9566d5dd4f22f6225` 的 [`MiniMaxH3Ref2VAResizeSetupStep.__call__`][4]：

```python
if set(kinds) == {"audio"}:
    raise ValueError("An audio reference cannot be used without at least one image or video reference.")
```

官方 README 的 Ref2VA input condition 寫「images, videos, and/or audio」，且架構寫 audio 僅經 AudioVAE，這與 Comfy 的能力方向一致；但 README 不能推翻某一 framework 的明確 guard。[5] h3.c 的限制也不是模型架構證明，而是其現行入口契約。[6]

## 2. 有序參考、標籤與影片替換音訊

### 模型語義

官方 prompt guide：[Picture N] 的實際文字格式是 `<Picture N>`，人物／商品可定義為 `<Subject N>` 並引用一張或多張圖片；`<Video N>` 可提供運鏡、剪輯、節奏或時間結構。只參考運鏡／動作時是 `reference generation`，不是自動成為 `video editing`。`<Audio N>` 可以是獨立音檔或啟用的影片 soundtrack；影像與音訊編號各自獨立。[10]

官方規格：圖片最多9張；影片最多3段，每段2–15秒、影片總長最多15秒；音訊最多3段，每段2–15秒、音訊總長最多15秒；混合素材最多12檔。輸出24 fps、32 kHz stereo，官方 H3-Base 輸出範圍4–15秒。[5]

### 各入口順序並不相同

- h3.c：reference flags 可重複，保留 CLI 實際順序；`--ref-image` 圖片、`--ref-silent-video` 無聲影片、`--ref-video` 保留 embedded soundtrack、`--ref-video-audio VIDEO AUDIO` 以明確音檔替換 soundtrack、`--ref-audio` 追加獨立音檔；不能和 first/last anchors 混用。[6]
- Comfy：節點固定先 images、再 videos、最後 standalone audios；每個影片的 soundtrack `<Audio j>` 放在該 `<Video k>` 前，型別內以1起算。影片 soundtrack 依 suffix 配對：`ref_video_audio_N` 對 `ref_video_N`，並非依 `<Audio j>` 與 `<Video k>` 數字相等配對。下游 tokenizer 則保留收到的 `ref_items` 順序。[1][2]
- Comfy 的 video input 是 frames，不是自動拆音的容器：要保留原音，外部 loader 必須把音軌接到對應 audio socket；要替換，只接替換音檔至同一 socket。它將 audio/video latent 作成同一 `video_audio` block，而非把替換音檔另加成不配對 standalone audio。[1]
- ModelTC JSON 文件說 `references` 是 ordered list；wrapper 按 list 順序 normalize。[4][7]

Comfy 程式還把 reference video 截到輸出 frame_count，再向下截成 `17k+5`；最低允許5幀（約0.2秒），與 schema tooltip 的2–15秒／官方模型規格不同。Qwen 影片觀察取2 fps，visual VAE 使用24 fps frames；不要拿 UI 可輸入範圍當作訓練品質保證。[1]

「替換參考 soundtrack」不等於最終輸出原樣 mux。官方 guide 明確區分 `fully_copy`、`partially_copy`、`reference`；若需求是最終音軌 bit-exact 沿用，僅依生成 prompt 無法提供此保證，需另行指定最終媒體處理契約。[10]

## 3. LoRA key families：實際載入契約

### Diffusers／PEFT

ModelTC [`inference_minimax_h3.py`][7] 使用：

- `transformer_blocks`，`token_refiner.refiner_blocks`；attention target `to_q`、`to_k`、`to_v`、`to_out.0`；FFN `ff.net.0.proj`、`ff.net.2`。
- factor suffix 明確是 `.lora_A.default.weight`、`.lora_B.default.weight`。
- 建立 `LoraConfig(r=rank, lora_alpha=lora_alpha, use_rslora=False)`；預設 alpha=8，報告有效縮放 `lora_scale * lora_alpha / rank`。

因此不能只把 `.lora_A.default.weight` 改成 `.lora_down.weight` 就當作完成格式轉換：還要 split/fuse QKV、module remap、保持 alpha/rank 和矩陣方向。

### Comfy／native

Comfy [`Attention`、`MLP`][3] 使用 fused `attn.qkv_proj`、`attn.out_proj`、`mlp.fc1`、`mlp.fc2`；這和本地 native inventory 對應。h3-lora 現行 [`h3_lora.c`](h3_lora.c) 的 `semantic` / `split_key`（約121–200行）接受：

- native stem 如 `blocks.0.attn.qkv_proj`；可剝除 `model.diffusion_model.` 或 `diffusion_model.`。
- 受控的 Kohya alias，如 `lora_unet_blocks_0_attn_qkv_proj`。
- `.lora_down.weight` / `.lora_up.weight` 或 `.lora_A.weight` / `.lora_B.weight`，另可有 `.alpha`。

**目前該列舉並不等於支援真正的 Diffusers split-QKV key family**，也不接受 ModelTC 的 `.default` suffix。應明示 unsupported，而不是默認名稱相似即相容。ModelTC 為同一 LoRA 提供 BF16（Diffusers）和 ComfyUI 兩份檔案，也佐證應選正確 release artifact，而非混用。[7][8]

本次沒有下載／逐個解析遠端 safetensors header，故未聲稱所有現存 Comfy 發行檔都使用完全相同 prefix、alpha metadata 或涵蓋所有 module；精確相容性仍應以實際檔案 inventory 為準。

## 4. Ref2VA Turbo8 設定與版本風險

本地 `tools/turbo_models.json` 已記錄 `minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors`：8步、res_multistep、simple、CFG1、strength1、1344×768、video/audio shift12/3。其 settings_note 明文說省略的 shifts **adopt Comfy H3 defaults**；因此是本地可重現設定，不是證實 Ref2VA8 訓練 shifts。Comfy `MiniMaxH3SigmaShift` 原始預設正是12/3。[1]

LightX2V 的第一手 [release announcement #57][9] 列出新 REF2V 8-step v1.1 BF16 與 ComfyUI artifacts，要求 inference step count 配合 checkpoint；video flow shift6 的建議**明確僅適用 FL2V，不適用 REF2V**。該公告沒有提供 REF2V 專屬數值 shift。不要靜默把 v1.0 換 v1.1，也不要把 FL2VA 768p 6/3 的建議套到 Ref2VA8。

ModelTC repository README 目前主要表格仍列舊 Ref2VA4 v0.1（544p、12/3），並推薦 reference resize `match`，與較新的 HF 公告不同步。`match` 為依目標 pixel area、只縮不放、保持比例、32-pixel rounding；能避免不必要 reference token 成本，但不證明 Turbo8 對純音訊的品質。[4][8][9]

## 5. h3-apple 的適用範圍

[h3-apple README][11] 主入口是1–9張圖加 prompt 生成5–15秒影音；可作 Apple 原生多圖流程參考。該 README 未提供 reference video 或純 reference audio 入口，因此不能用它作為這兩項能力已完成的證據。其輸出有聲也不等於輸入音訊可用。

## 6. 尚未被本研究證明的事項

- Comfy 純音訊路徑已存在，但此研究沒有實測音色、節奏／唇形同步、人物／商品一致性，亦沒有證明任一 Ref2VA Turbo LoRA 的訓練資料含純音訊。
- ModelTC guard 的移除是否足夠使該特定 Diffusers 版本端到端正常，未證明；應以 Comfy 的完整 presentation、tags、packed layout 與 audio latent 契約為移植依據，而非只刪 guard。
- 沒有找到第一手 REF2V8 v1.0 專屬 training shift 數值；12/3 必須註記為採用框架預設。
- 修改前，h3-lora 入口禁止純音訊、active Ref2VA LoRA 要求實際圖片；主代理已透過真實 CLI 重現兩種拒絕。下節記錄本次改接後的實際結果，不再將這些舊限制當成目前契約。

## 第一手來源

[1]: https://github.com/Comfy-Org/ComfyUI/blob/7cebca52227c63b4180ceed448cbd7ad22903b6d/comfy_extras/nodes_minimax_h3.py
[2]: https://github.com/Comfy-Org/ComfyUI/blob/7cebca52227c63b4180ceed448cbd7ad22903b6d/comfy/text_encoders/minimax.py
[3]: https://github.com/Comfy-Org/ComfyUI/blob/7cebca52227c63b4180ceed448cbd7ad22903b6d/comfy/ldm/minimax/model.py
[4]: https://github.com/ModelTC/Minimax-H3-Turbo/blob/02e26d591f7a04d5d1a074c9566d5dd4f22f6225/minimax_h3_ref2va_pipeline.py
[5]: https://github.com/MiniMax-AI/MiniMax-H3/blob/d21241f0a4b3acbb34c97dae47fa417b7065e438/README.md
[6]: https://github.com/antirez/h3.c/blob/main/README.md
[7]: https://github.com/ModelTC/Minimax-H3-Turbo/blob/02e26d591f7a04d5d1a074c9566d5dd4f22f6225/inference_minimax_h3.py
[8]: https://github.com/ModelTC/Minimax-H3-Turbo/blob/02e26d591f7a04d5d1a074c9566d5dd4f22f6225/README.md
[9]: https://huggingface.co/lightx2v/Minimax-h3-Turbo/discussions/57
[10]: https://github.com/MiniMax-AI/MiniMax-H3/blob/d21241f0a4b3acbb34c97dae47fa417b7065e438/skills/h3-prompt-writing/references/ref-en.txt
[11]: https://github.com/RhinoQ/h3-apple/blob/main/README.md

## h3-lora 本次接線與實測結果

變更：`h3.c` 移除 active Ref2VA LoRA 強制有圖片及 match sizing 的額外限制；共用有序參考驗證仍保留型別、數量、音訊長度、first/last anchors 互斥、LoRA target 一致與 SSD streaming 限制。純音訊不再被 visual-only guard 阻擋，且實際進入 Ref2VA presentation；`h3_multimodal.c` 建立 `<Audio N>: ` + prompt，沒有視覺 spans 時使用既有純文字 Qwen encoder。音訊仍經原有 AudioVAE、condition augmentation、packed layout 與 DiT；沒有 dummy image、假 latent 或事後將輸入音訊冒充生成音訊。

真實推論主機為 Apple M5 Max 128 GiB。使用既有 Ref2VA base 與 `minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors`，SHA-256 `6a56f41ab4229c9dd845b9501bbd475ee57e112d846cf2e819d534a1ae928c5a`，強度1、RES/simple、8步、shifts12/3、50層、reuse/core-reuse1、seed42。**512×512 僅供接線 smoke，並非此768p權重的正式畫質建議。** 來源人物／影音取自既有生成短片；商品圖是明確標示的合成紅盒測資，不是使用者真實人物／商品。

| 實跑組別 | 原始幀數／24fps | 含載入 wall 秒數 | 去噪秒數 |
|---|---:|---:|---:|
| 01 多圖：人物＋商品，match | 39 | 84.347 | 14.811 |
| 02 無聲影片動作／運鏡參考 | 39 | 110.717 | 20.930 |
| 03 純音訊參考 | 39 | 79.794 | 13.081 |
| 04 人物圖片＋獨立音訊 | 39 | 86.441 | 14.815 |
| 05 影片含原音 | 56 | 143.768 | 35.324 |
| 06 影片配對替換音訊 | 56 | 140.199 | 36.644 |
| 07 無聲影片＋獨立音訊 | 39 | 112.344 | 22.497 |
| 08 多圖：人物＋商品，max | 39 | 84.521 | 14.697 |

八組均產生真實 MP4；每組引擎記錄208組已載入的 LoRA pairs/targets，sampler trace 實際 NFE8、success=true，影像512×512／24fps／對應原始幀數，音訊32kHz stereo。原始影音在遠端完整解碼後，所有成品、輸入測資、紀錄與參數複製至使用者 Mac；本機重新核對所記錄 SHA-256，八支原片與八支一秒預覽均完整影音解碼成功。

一秒預覽為原片前24幀及音訊以CRF16/AAC重編；39幀原片1.625秒，56幀原片2.333秒。配對影片 soundtrack 至少2秒，故05/06必須用較長原片；表中耗時不是只生成一秒的耗時，不含後處理與傳輸。

四項負向 CLI 情境均實際拒絕：FL2VA LoRA target 搭 ordered refs、active LoRA 搭 SSD、ordered refs 搭 first-frame anchor、22幀影片搭 soundtrack（不足2秒）。編譯與既有 `h3_lora_tests`、`h3_tests`（1768 checks）、`h3_sampler_tests` 通過。未增加實作／文字比對測試以重複宣稱接線成功。

抽樣檢視0／12／23幀：多圖組可見人物持紅盒；影片組可見連續走路姿態；純音訊組產出公園場景。多圖紅盒在初始幀尚未持在手上，不能宣稱逐幀商品一致性。**沒有主觀聽音／音畫同步評分、運鏡軌跡量測、人物辨識或商品保真度驗證，也未聲稱任意FL2VA風格LoRA都與Ref2VA訓練相容。** 同seed、同畫布但輸入結構與原片幀數不同，耗時僅是逐組單次觀察。

## 2026-10-26：參考圖＋參考聲音＋首圖條件的上游求證

本節為網路第一手文件／程式靜態核對，未執行上游推論，也未改動本機或128G生成程式。結論：**不能從 h3-lora 的互斥檢查推論模型本質上不支援；上游已有 references＋時間定位 keyframe guide 的組合路徑。** 但入口可接、checkpoint品質與逐像素首幀保證是不同層級。

- [MiniMax hosted API V2](https://platform.minimax.io/docs/api-reference/video-generation-v2-create.md) 的 `VideoGenerationV2Req.content` 明文禁止 `reference_image/reference_video/reference_audio` 與 `first_frame/last_frame` 混用。此為託管API契約，不代表所有自架框架都禁止。
- [官方 Ref2VA prompt guide](https://github.com/MiniMax-AI/MiniMax-H3/blob/35491cdba2adfe62a510f725e8619f8e58783ea2/skills/h3-prompt-writing/references/ref-en.txt) §2.2允許 `<Picture N>` 作首幀／關鍵幀／尾幀，§2.4定義音色參考，§3允許 `keyframe completion` 等任務相加。這證明有語義表達方式，不是固定輸出latent或逐像素相同的保證。
- [ComfyUI nodes](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_minimax_h3.py) 的 `MiniMaxH3ReferenceToVideo` 可建立圖／音 `minimax_refs`，再串 `MiniMaxH3AddGuide(frame_idx=0)` 追加 `minimax_keyframes`；[MiniMaxH3.extra_conds](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/model_base.py) 與 [PackedLayout](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/minimax/model.py) 同時處理兩者。這是實際編碼條件組合，不只是prompt描述。Comfy連結為查核當下移動分支。
- [SGLang cookbook §2](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3) 明文列出 Ref2VA references 可與首尾keyframes結合；固定revision `ee98bb09b02877ce55bbacd304121dfb7a2692a9` 的 [task_profiles.py](https://github.com/sgl-project/sglang/blob/ee98bb09b02877ce55bbacd304121dfb7a2692a9/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/task_profiles.py)、[request_validation.py](https://github.com/sgl-project/sglang/blob/ee98bb09b02877ce55bbacd304121dfb7a2692a9/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/request_validation.py) 與 [packed_sequence.py](https://github.com/sgl-project/sglang/blob/ee98bb09b02877ce55bbacd304121dfb7a2692a9/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/packed_sequence.py) 有混合驗證與packing實作。profile註解保留「for hybrid checkpoints」限定；不能由接受輸入便宣稱目前官方Ref2VA＋Turbo品質已驗證，也不能把這與必須使用合併權重的 `--model-variant hybrid` 畫等號。
- 上述guide以獨立固定conditioning rows引導，target rows仍另行去噪；不等於逐步覆寫輸出首幀latent。**目前沒有證據要求先重訓，但也沒有本次實跑證據保證現有Ref2VA／Turbo權重的首圖、人物及聲音遵循品質。**

工程方向：以ComfyUI／SGLang既有混合conditioning作移植依據，修改本機首圖編碼、layout／RoPE、condition rows及快取契約，不能只刪互斥檢查；先驗證現有base，再分開驗證Turbo。無須先假定必須改模型或換權重。
