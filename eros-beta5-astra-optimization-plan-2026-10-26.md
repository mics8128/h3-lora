# Eros beta5：三個 Astra 視角的長序列加速決策

## 狀態與結論

依使用者要求，`profiles/h3.json`已選快速4步ER-SDE/beta57、M5原生INT8投影、BF16影片VAE，保留768×1344/24fps、50層、reuse/core1、完整素材與beta5 F32final。實際離線prepare_video已走雙角色5圖2音訊、141幀及步數影響render-key檢查；未送SSH或推論。影片生成仍暫停。

後續品質保守實作與非影片量測已完成：FC2 K1024未建立收益，PR77式dense query切塊亦未建立穩定有效加速；兩者未採用且active隔離來源已還原，正式部署不變。見[FC2否決證據](eros-beta5-quality-safe-fc2-validation-2026-10-26.md)、[dense query數值與速度](eros-beta5-quality-safe-dense-validation-2026-10-26.md)、[最佳已知設定](eros-beta5-best-settings-quality-safe-2026-10-26.md)。下文保留當時提案與未執行選項，不把早期排序當成現在已證實的速度推薦；sparse/W4A8/跨步重用在後續品質契約下排除本輪採用。

目前結案與Luna全公開分支metadata補查、Astra固定來源增量評估、串流／最佳設定整合，統一看[品質優先最終結論](eros-beta5-quality-safe-final-plan-2026-10-26.md)。該結論優先於下文早期提案排序；沒有將沒有收益或未驗收的候選部署為正式預設。

三份獨立分析均實際透過`astra_consult`明確路由`llm-esapp/gpt-6-astra`，唯讀檢視已安裝來源，未替代為普通代理自行結論，未修改原生核心、編譯或跑測試／benchmark。不同側面不是三次實測。

- [Attention與演算法](eros-beta5-astra-attention-2026-10-26.md)
- [INT8、MLP與融合kernel](eros-beta5-astra-quant-fusion-2026-10-26.md)
- [執行調度與記憶體生命週期](eros-beta5-astra-execution-2026-10-26.md)

## 共同否定的錯誤捷徑

現有約413秒denoise wait包含實際GPU/MPSGraph child work，不是可直接刪的CPU空轉；root-gpu約2.7秒不是GPU總時間。現有activation已預配置重用、denoise新配置為0；M5 INT8融合也已啟用，不能重複當新優化。原生完整attention是MPSGraph SDPA，內部workspace算法仍未知。保留LLM權重的RAM預算不等於同時重GPU推理不競爭。

141幀target自身有42,336個video tokens＋470個audio rows，共42,806，還需加文字與全部reference/guide；雙角色不是batch翻倍。新5圖2音訊的實際seq_len須在之後獲准實驗記錄，不能只測2048tokens後宣稱正式長序列成功。詳細公式與引用見Attention筆記。

## 整合排序：兩條主線，一條實驗線

| 優先 | 提案 | 分類 | 決策與必要證據 |
|---|---|---|---|
| 0 | 長序列完整GPU timeline，拆SDPA/QKV/out/MLP與framework memory | 必要歸因，不是加速本身 | 記實際seq_len與segment spans；不以root timestamps猜瓶頸；停在此處不稱已優化 |
| 1A | 原生Metal dense tiled online-softmax，保持完整joint K/V | 同模型語義、浮點可能不同 | 激進主線；先32×64/32×128tiles，F32softmax/accumulator，head-major輸出；若SDPA非熱點則降優先 |
| 1B | FC2 K1024整scale-group TensorOps＋逐層權重預封裝 | 現行INT8數值精確候選 | 與1A不同熱點；先不變group/F32累加次序；不永久留整模型雙份INT8；若MLP非熱點則降優先 |
| 2 | FC1→SwiGLU→groupquant真正融合、Q/K片上norm/RoPE epilogue | 現行INT8數值精確候選 | 減全序列中間張量搬運；必處理alias讀寫競爭，保留BF16rounding邊界；各自獨立驗收 |
| 3 | Rectangular query SDPA adapter／bounded command windows | 完整dense／排程候選 | 小型判別實驗；每query仍看全KV；計入gather/scatter與dispatch，不保證MPSGraph矩形shape變快 |
| 4 | Split-KV online-softmax＋合併 | 完整dense，歸約順序可能不同 | 只有1A量到occupancy/長迴圈問題才做；partial scratch可達數GiB，受40GiB目標約束 |
| 實驗A | 中段step/layer target-video局部或稀疏attention，其餘AV/text/reference仍dense | 明確品質近似 | 最激進算量路徑；不是作者推薦、不預設；保留單一softmax、原RoPE/segment，不以不同modality各自softmax取代joint |
| 實驗B | FC2-only W4A8 | 明確品質近似 | 理論FC2 payload省約1.79GiB，並不減INT8MAC；unpack可能變慢，不能先稱加速 |
| 後置 | GPU-state ER-SDE／persistent worker | 需大工程但直接收益證據弱 | CPU sampler不占主要已知wall；常駐主要省準備，不用它取代核心優化 |

1A與1B是可獨立研究的熱點分支，不預設全部收益相乘；正式大改只在歸因與各自microkernel/完整block證據後整合。

## 真正激進但受控的稀疏方案

僅在明確研究profile啟用：evaluation1/4全dense，evaluation2/3部分中段layers的target-video↔target-video改時間/空間局部，其餘text/reference/guides/targetaudio雙向連接保留。local鄰近依原model-time/spatial，不依packed-row距離；保持原positions與素材順序。不新增reference block-diagonal mask冒充原dense語義。

這會改模型：global refresh無法保證恢復身份、遮擋後重現、遠距動作及AV同步；必須多人物/多reference完整媒體驗收。reference hidden每層受target影響，跨denoise快取reference K/V也屬近似，不是安全不可變cache。step residual reuse因作者reference警告及四步本就很少，只保留次級研究選項。

## 記憶體與收益目標

影片程序約40GiB軟目標，小幅超過需事先明確揭露，不自動制定新硬上限。優先用於有限tile/in-flight scratch；不把既有26.3GiB樣本與40GiB之差全吃掉。禁止N×Nmask/score全量常駐、全模型BF16+INT8+packed多份、雙重完整context盲目常駐。分別記process footprint、Metal與framework storage及系統pressure/swap；不能把不同口徑相加。

既有4步總489.602秒、去噪413.015秒（84.36%）、其他76.587秒。若整段去噪假設快2倍，算術約283.094秒；快4倍約179.841秒。**只是Amdahl條件式計算，不是任何提案預測或極限驗收。** Attention/MLP各自占比尚未知，不能承諾單一kernel就做到這些數字；此樣本即使去噪歸零也不能達60秒。

## 驗收與實作次序

1. 待使用者允許恢復測量後，在隔離runtime clone固定4步、實際正式長序列、多reference，記完整GPU timeline；不停止別人服務，不重送未知job。
2. 同輸入QKV測dense attention候選；同輸入INT8/scales測FC2/fusion候選。新SDPA對獨立高精度oracle與MPSGraph做事前訂定誤差門檻，不能失敗後才放寬；數值精確fusion以目前INT8逐位比對。
3. 不改projection的attention、及不改attention的MLP分開ABBA，計首次compile/steady-state、完整block/denoise、記憶體；收益不足刪掉候選，不堆永久旗標。
4. 連接單層→單eval→固定四步完整影音；驗人物/說話者綁定、中文台詞、音色、嘴型/AV同步、轉身遮擋與素材A→B→A隔離；full AV decode/hash只證明技術完整。
5. 排程/lifetime/kernel需保持resource在completion後才回收，beta5rank8/F32final契約不變；如修改sampler必補state/history/RNG與pause/resume既有語義驗證。
6. exact語義快路徑通過後才考慮預設，能力不符在dispatch前退回MPSGraph/原kernel；GPU錯誤不在未知狀態靜默再生成。Sparse/W4A8永遠獨立、明確opt-in，不與降解析度/少素材混合宣稱無損。

目前交付是設定採用與工程設計，不是核心實作或效能承諾；原生部署未更動，沒有新增影片。
