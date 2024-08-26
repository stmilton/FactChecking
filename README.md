# VeriScan
## Intro
這是一個用於快速事實查核的工具，前端應用 Chrome 的擴充功能，後端應用 FastAPI 串接基於 Transformer 架構的預測模型。
> work status: Building API

## Environment
- Python: 3.9.18

## Usage
### 建置 API
請按照下列步驟進行前置作業
1. 下載模型的 [ckeckpoint](https://drive.google.com/file/d/1MXZZRVMeOMAia0OGlXIsZYVb_Sp1pI2r/view)。
2. 將解壓縮後的 ckeckpoint 資料夾，置放於 FactVerificaionAPI 資料夾裡面
![](https://i.imgur.com/BnDuh7N.png)
3. cd 至 FactVerificationAPI 資料夾
4. 執行 `uvicorn main:app --host 0.0.0.0 --port 80`

### 使用 API
> Label number 對照 -> {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2}
1. equery 範例
    - claim: string
```python
claim = "Jackie (2016 film) was directed by Peter Jackson."
web = requests.get(f"http://140.115.54.36/equery/?claim={claim}")
print(web.text)
# result -> {"equery":"Jackie_(2016_film)","state":"1"}
```
2. everify 範例
    - claim: string
    - url: string
```python
claim = "Jackie (2016 film) was directed by Peter Jackson."
url = "https://en.wikipedia.org/wiki/Jackie_(2016_film)"
web = requests.get(f"http://140.115.54.36/everify/?claim={claim}&url={url}")
print(web.text)
# result -> {"label":1,"evidence":
#   ["Jackie grossed $14 million in the United States and Canada and $22.",
#   " Thus, Jackie is the first film that he could approach from a woman's perspective." ...],
#   "state":"1"}
```
3. cverify 範例
    - claim: string
    - url: string
```python
# cverify
web = requests.get(f"http://140.115.54.36/cverify/?claim={claim}&url={url}")
print(web.text)
# result -> {"label":0,"evidence":["　　杨代办强调，事实一再证明，对中方的指控...],"state":"1"}
```