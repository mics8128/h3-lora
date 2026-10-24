# 模型檔案：目前保留狀態與清理紀錄

主機：`ted`；根目錄：`/Users/tedmacbookpro/projects/h3-models`。依使用者明確指示，已刪除4個額外風格LoRA及Eros轉換前、後的2個權重檔。**其餘模型全部保留，沒有搬移、更名或縮減Turbo比較套件。** 日常用法見[README](README.md)，完整控制項見[RUNTIME_REFERENCE](RUNTIME_REFERENCE.md)。

## 1. 這次實際刪除

完成時間：2026-10-24T16:17:06Z（台灣時間2026-10-25）。只對下列6個regular files逐檔unlink，沒有遞迴刪除目錄，也沒有跟隨共用symlink。

| 相對於模型根目錄的路徑 | 刪除前bytes |
|---|---:|
| `loras-style/minimalisticStyle_v1.safetensors` | 310,170,064 |
| `loras-style/physics/wushu_spatial_physics_clean_3000_pruned.safetensors` | 155,109,672 |
| `loras-style/studio1939/studio1939-light.safetensors` | 65,628,096 |
| `loras-style/studio1939/studio1939-strong.safetensors` | 262,433,016 |
| `Eros-Max-beta6/source/10Eros_Max_h3_8stepTURBO_beta6_bf16.safetensors` | 66,246,043,136 |
| `Eros-Max-beta6/FL2VA/transformer/model.safetensors` | 66,280,487,808 |
| **合計** | **133,319,871,792** |

移除的檔案logical size共**133.320GB／124.164GiB**。這不是已保證釋出的磁碟空間：刪除後立即讀取的`statvfs`可用容量未增加，該次觀察反而少約19.49MB；主機同時有其他活動。唯讀檢查發現Data volume有**21個Time Machine本機快照**，均標示purgeable。**[INFERENCE]** 快照或APFS延遲回收可能仍保留資料區塊；沒有證明各檔案的實際reclaimed bytes，也沒有刪除快照、清Time Machine或更動備份設定。

Eros的轉換腳本、header、生成log及symlink保留作紀錄，直接regular-file大小僅196,872bytes；**Eros已不是可供推論的完整模型root**，不應再用它執行生成。`loras-style/`剩82,596bytes下載／來源metadata，4個權重均已刪除；保留metadata不代表模型仍存在。

## 2. 其餘全部保留

| 集合 | 直接持有的不同inode logical bytes | 處置 |
|---|---:|---|
| `MiniMax-H3/` | 144,051,191,629 | 保留：FL2VA及共用組件 |
| `MiniMax-H3-Ref2VA/` | 66,280,526,873 | 保留：Ref2VA獨立transformer；其餘共享 |
| `loras-turbo/` | 27,492,408,486 | **21個Turbo全部保留**，不是只留日常3個 |
| `loras/` | 918,543,562 | 保留Unlocked及原有Larry hardlink alias |
| `loras-style/` | 82,596 | 僅剩metadata |
| `Eros-Max-beta6/` | 196,872 | 僅剩轉換／生成紀錄與共用連結，無Eros權重 |

跨列相加要扣除Larry共用的620,285,592bytes inode。清理後總計**238,122,664,426bytes／221.769GiB**（不同regular-file inode去重），260個regular-file路徑、259個不同inode、10個symlink。`st_blocks×512`合計238,123,429,888bytes，仍不能等同APFS容器實際獨占空間。

之前提出的「封存另外18個Turbo」**本次沒有執行**，使用者已選擇其餘保留。完整21檔的名稱、來源、原始SHA紀錄與參數仍在[tools/turbo_models.json](tools/turbo_models.json)；這輪沒有重讀全部權重或重新計算其內容SHA。

### 日常3個入口，不限制保留集合

| 用途 | `loras-turbo/`內檔名 | 起點 |
|---|---|---|
| 544p草稿 | `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | 960×544、Euler/simple、4步 |
| 768p正式 | `minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors` | 1344×768、RES/simple、5步 |
| Ref2VA | `minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` | RES/simple、8步 |

三者各1,956,193,000bytes；strength1、shifts12/3是已揭露的採用設定，不宣稱每版最佳。v1.2檔名的4step不代表日常建議4步。可選風格`loras/Minimax_H3_Unlocked_V2.1.safetensors`（298,257,970bytes）**未刪除**；它不是本次清除的4個額外風格之一。

## 3. 共用依賴與hardlink仍完整

- `MiniMax-H3-Ref2VA/FL2VA`指向`MiniMax-H3/FL2VA`整棵樹。
- Ref2VA下的`audio_vae`、`text_encoder`、`tokenizer`、`video_vae`指向官方FL2VA同名組件。
- Eros留下的4個共享component symlink及transformer config symlink也只作紀錄；其目標均保留。
- [`h3.c`](h3.c)目前在建立context時仍inventory FL2VA組件，即使只做Ref2VA，也不能刪FL2VA樹或只留下config/index而移除shards。
- text encoder14個shards、各官方transformer13個shards與原始video/audio VAE均保留。BF16 VAE是runtime精度選項，不是另一份權重；LightVAE尚未實作。
- Larry的`loras/minimax_h3_turbo_v4_step600_pruned_comfyui.safetensors`與`loras-turbo/`同名檔仍共用同一inode。只刪其中一個名字不會回收這份權重；本次兩者都保留。

## 4. 實際驗證與證據

刪除前再次核對6檔device/inode、size、mode、mtime及nlink=1，且所有父目錄不是symlink。執行前比較整個model root的regular files／symlink metadata與preflight一致，才允許unlink。

刪除後確認恰好少這6個檔案；其他全部regular-file metadata（inode／size／mtime／mode／link count）及symlink字串／解析目標與刪除前相同。這是metadata比對與精確操作範圍證據，不是重新hash數百GB權重內容。

刪除後在`ted`實際執行兩個官方root的`./h3 --info -d ROOT`，均成功：FL2VA13個transformer shards；Ref2VA root額外含13個Ref2VA shards；兩者共享text encoder14檔、video VAE及audio VAE。這是header／metadata smoke，**沒有重新生成影片或宣稱做過完整推論驗收**。

`lsof`沒有列出6個目標的open handles，但對不相關的SMB Time Machine mount發出stat警告；警告完整保存在preflight紀錄，不將其描述成全系統open-file查核無警告。該SMB device與模型所在device不同。

本機清理證據：`/Users/mics/Movies/H3-Model-Cleanup-20261025/`：

- `plan.json`：刪除前快照、精確6檔身份、open-file警告與原始容量。
- `removed.json`／`result.json`：逐檔unlink紀錄、操作後metadata比對與容量觀察。
- `after-inventory.json`：目前保留檔案及symlink metadata。

清理前原始盤點仍在`/Users/mics/Movies/H3-Model-Inventory-20261024/`；其中`inventory.json`與`summary.json`是**歷史快照，不是目前清單**。這次沒有建立權重備份；來源檔亦已依使用者授權刪除。任何再刪除、移動剩餘模型或處理APFS／Time Machine快照，都需另行明確授權。
