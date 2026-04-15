# 強化學習作業：Cliff Walking 環境下的 Q-learning 與 SARSA 比較

本專案實作並比較了兩種經典強化學習演算法——**Q-learning**（離策略, Off-policy）與 **SARSA**（同策略, On-policy），在經典的「懸崖行走（Cliff Walking）」環境中的學習行為、收斂特性及策略差異。

## 專案結構
- `main.py`: 實作了 $4 \times 12$ 的 Cliff Walking 網格環境、Q-learning 與 SARSA 演算法，以及訓練與圖表繪製的主要程式碼。
- `cliff_walking_rewards.png` (執行後自動產生): 顯示兩種演算法在 500 回合內的累積獎勵收斂曲線（取 50 次實驗平均）。
- `cliff_walking_policies.png` (執行後自動產生): 視覺化兩種演算法最終學習到的行動策略路線（箭頭指示）。

## 環境設定與參數
- **環境**: $4 \times 12$ 網格，起點位於左下角 `(3, 0)`，終點位於右下角 `(3, 11)`，底部邊緣 `(3, 1)` 到 `(3, 10)` 為懸崖（Cliff）。
- **獎勵機制**: 每走一步得 `-1`，掉入懸崖得 `-100` 並退回起點。到達終點結束該回合。
- **學習率 (Alpha, $\alpha$)**: `0.5`
- **折扣因子 (Gamma, $\gamma$)**: `1.0`
- **探索機率 (Epsilon, $\epsilon$)**: `0.1` (採用 $\epsilon$-greedy 策略)
- **訓練回合數**: `500`

## 安裝與執行指引

### 1. 安裝依賴套件
在執行前，請確保您的 Python 環境（建議使用 Python 3.7 以上版本）已安裝 `numpy` 與 `matplotlib`。您可以透過以下指令安裝：
```bash
pip install numpy matplotlib
```

### 2. 執行程式碼
使用終端機（或 PowerShell）進入該資料夾，並執行 `main.py`：
```bash
python main.py
```
執行過程約需數秒鐘，程式會在背景執行兩種演算法各 50 次以進行平均。執行完成後，終端機會提示 "Experiment completed. Outputs saved."，並在同目錄下產生結果圖表。

## 實驗結果簡述
- **SARSA (On-policy)**：傾向學習「安全」的路徑。為了避免在隨機探索 ($\epsilon=0.1$) 時落入懸崖帶來巨大的 `-100` 懲罰，SARSA 會選擇繞遠路走在網格上方，因此在訓練過程中能穩定獲得接近 `-25` 的平均累積獎勵。
- **Q-learning (Off-policy)**：傾向學習「理論最優」的路徑。它會勇敢地貼著懸崖邊緣走捷徑。但由於 $\epsilon$-greedy 探索的介入，每次靠崖行走時有 $10\%$ 的機率隨機移動而踩空落下，這使得它在訓練過程中非常容易失敗，平均累積獎勵只能停留在 `-45` 左右，震盪也更為劇烈。
