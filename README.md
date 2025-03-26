# Gold-Price-Prediction-with-Random-Forest
# -*- coding: utf-8 -*-
"""
@author: linruibin
"""

#X:非黃金類指標/ Y:黃金類指標(使用時機:當日開盤後)
#使用非黃金類指標漲跌量，對應黃金類指標漲或跌訓練模型
#學習非黃金類指標的價格變化與黃金相關股票漲跌之間的關係。
#通過對非黃金類指標的當日價格變化進行標準化和分類，捕捉到市場中不同程度的價格波動對黃金價格的影響。

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
import yfinance as yf

#===========取得歷史資料集======================
#定義股票代碼及其對應變量名稱
tickers = {
    'USO': 'USO',#油 
    'PALL': 'PALL',#鈀金 
    'SLV': 'SLV',#銀 
    'PLG': 'PLG',#鉑金 
    'ZB=F': 'ZB=F',#美國國債期貨
    '^GSPC': 'S&P 500',#標普500指數 
    'EURUSD=X': 'EUR/USD',#歐元美元匯率 
    'AGZ': 'AGZ',#iShares機構債券ETF 
    'TIP': 'TIP',#iShares抗通脹債券ETF 
    'ALI=F': 'ALI=F',#鋁期貨 
    
    #債券
    #'UB=F': 'UB=F',#超長期美國國債期貨
    #'ILTB': 'USD Bond',#美國債券 
    #'BOND': 'BOND',#PIMCO主動債券ETF
    #'TN=F': 'TN=F',#超長期10年期美國國債票據
    #'FLOT': 'FLOT',#iShares浮動利率債券ETF
    #'HYG': 'HYG',#iShares高收益公司債券ETF
    #'MBB': 'MBB',#iShares抵押支持證券ETF 
    #'MUB': 'MUB',#iShares全國市政債券ETF 
    #金屬
    #'HG=F': 'HG=F',#銅期貨
    #'MOON.V': 'MOON.V',#藍月鋅業公司
    #'IBG.AX': 'IBG.AX',#鐵皮鋅業有限公司
    #'SANDUMA.BO': 'SANDUMA.BO',#桑杜爾錳和鐵礦有限公司
    
    #備用X
    #(美元貶值，商品價格漲)(降息刺激消費)
    #'DBC': 'Invesco DB Commodity Tracking',#大宗商品ETF，包含能源、貴金屬、工業金屬和農產品等
    #'GSG': 'iShares S&P GSCI Commodity-Indexed Trust',#大宗商品ETF，涵蓋能源、工業金屬、貴金屬和農產品等 
    #'CORN': 'Teucrium Corn ETF',#玉米期貨，追蹤芝加哥期貨交易所的玉米期貨價格
    #'COMB': 'GraniteShares',#涵蓋 23 種大宗商品
    #'GCC': 'WisdomTree Continuous Commodity ETF	',#涵蓋能源、金屬和農產品等多種大宗商品
    #'FTGC': 'First Trust Global Tact Cmdty Strat ETF',#包括能源、工業金屬、農產品和貴金屬
    'DJP': 'DJP',#涵蓋能源、農產品、貴金屬和工業金屬
    #'BCI': 'abrdn Bloomberg All Commodity Strategy K-1 Free ETF	',#各類大宗商品，從能源到農產品
    
    #黃金
    #'GDX': 'GM_ETF',
    #'EGO': 'Eldorado',
    #'GLD': 'SPDR',
    'SGOL': 'SGOL',
    'PHYS': 'Sprott',
    'IAU': 'iShares'
}

#初始化數據框列表，下載數據
data_frames = []
for ticker, prefix in tickers.items():
    data = yf.download(ticker, period='10y')
    data.columns = [f'{prefix}_{col}' for col in data.columns]
    data_frames.append(data)
#合併DataFrame
combined_df = pd.concat(data_frames, axis=1)

#===========新增資料集內容以便計算======================
#為每個股票創建'previous_close'和'price_change'欄位
for prefix in tickers.values():
    combined_df[f'{prefix}_previous_Close'] = combined_df[f'{prefix}_Close'].shift(1)
    combined_df[f'{prefix}_previous_Low'] = combined_df[f'{prefix}_Low'].shift(1) #close改low
    combined_df[f'{prefix}_price_change'] = combined_df[f'{prefix}_Open'] - combined_df[f'{prefix}_previous_Low'] #close改low
#iShares收盤價變動
combined_df['iShares_close_change'] = combined_df['iShares_Close'] - combined_df['iShares_previous_Close']
combined_df['Sprott_close_change'] = combined_df['Sprott_Close'] - combined_df['Sprott_previous_Close']
combined_df['SGOL_close_change'] = combined_df['SGOL_Close'] - combined_df['SGOL_previous_Close']

combined_df['target'] = (combined_df['iShares_close_change'] > 0).astype(int)
combined_df['target4'] = (combined_df['Sprott_close_change'] > 0).astype(int)
combined_df['target5'] = (combined_df['SGOL_close_change'] > 0).astype(int)

#定義需要排除的黃金股票代碼
exclude_stocks = {'iShares', 'SGOL', 'Sprott'}

#定義包含非黃金股票的DataFrame
non_gold_low_columns = [f'{prefix}_Low' for prefix in tickers.values() if prefix not in exclude_stocks] #close改low
#創建包含非黃金股票收盤價的DataFrame
combined_df_full = combined_df[non_gold_low_columns] #每一天最後價格(當作預測時的收盤價格)
combined_df_full.dropna(inplace=True) 

#處理nan
combined_df.dropna(inplace=True)

#===========定義X,y並切割訓練測試======================
#創建特徵列，排除指定的股票
feature_columns = [f'{prefix}_price_change' for prefix in tickers.values() if prefix not in exclude_stocks]
X = combined_df[feature_columns]
y = combined_df['target']
y4 = combined_df['target4']
y5 = combined_df['target5']

#標準化和數值分類
#引入StandardScaler進行標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#定義數值分類函數
def classify_value_X(x):
    if x > 3:
        return "狂漲"
    elif x > 2:
        return "大漲"
    elif x > 1:
        return "漲"
    elif x > 0:
        return "小漲"
    elif x < -3:
        return "狂跌"
    elif x < -2:
        return "大跌"
    elif x < -1:
        return "跌"
    else:
        return "小跌"

#對標準化後的特徵進行分類
X_classified = pd.DataFrame(X_scaled, columns=feature_columns)
X_classified = X_classified.applymap(classify_value_X)

#進行One-Hot編碼
X_encoded = pd.get_dummies(X_classified)

#繼續原本的資料分割
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=777)
_, _, y4_train, y4_test = train_test_split(X_encoded, y4, test_size=0.2, random_state=777)
_, _, y5_train, y5_test = train_test_split(X_encoded, y5, test_size=0.2, random_state=777)



#定義超參數網格
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [6, 8, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': [None, 'balanced']
}

#超參數調整
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=777), param_grid=param_grid, scoring='f1', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_

#超參數調整
grid_search4 = GridSearchCV(estimator=RandomForestClassifier(random_state=777), param_grid=param_grid, scoring='f1', cv=5, n_jobs=-1)
grid_search4.fit(X_train, y4_train)
best_params4 = grid_search4.best_params_

#超參數調整
grid_search5 = GridSearchCV(estimator=RandomForestClassifier(random_state=777), param_grid=param_grid, scoring='f1', cv=5, n_jobs=-1)
grid_search5.fit(X_train, y5_train)
best_params5 = grid_search5.best_params_



#創建和訓練隨機森林模型
clf = RandomForestClassifier(**best_params, random_state=777)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
misclassified_samples = (y_test != y_pred).sum()
test_f1 = f1_score(y_test, clf.predict(X_test))
print("iShares最佳參數組合：", best_params)
print("iShares misclassified samples :", misclassified_samples, "筆") 
print("iShares Test accuracy =", clf.score(X_test, y_test))
print("iShares f1_score :", test_f1)

clf4 = RandomForestClassifier(**best_params4, random_state=777)
clf4.fit(X_train, y4_train)
y4_pred = clf4.predict(X_test)
misclassified_samples4 = (y4_test != y4_pred).sum()
test_f1_4 = f1_score(y4_test, clf4.predict(X_test))  
print("Sprott最佳參數組合：", best_params4)
print("Sprott misclassified samples :", misclassified_samples4, "筆")
print("Sprott Test accuracy =", clf4.score(X_test, y4_test))
print("Sportt f1_score :", test_f1_4)

clf5 = RandomForestClassifier(**best_params5, random_state=777)
clf5.fit(X_train, y5_train)
y5_pred = clf5.predict(X_test)
misclassified_samples5 = (y5_test != y5_pred).sum()  
test_f1_5 = f1_score(y5_test, clf5.predict(X_test))
print("SGOL最佳參數組合：", best_params5)
print("SGOL misclassified samples :", misclassified_samples5, "筆")
print("SGOL Test accuracy =", clf5.score(X_test, y5_test))
print("SGOL f1_score :", test_f1_5)


#===========取得最新數據======================
#下載最新的數據並計算price_change，僅計算非黃金股票
latest_prices = {}
for ticker, prefix in tickers.items():
    if prefix not in exclude_stocks:
        data = yf.download(ticker, period='1d', interval='1m', progress=False)
        if not data.empty:
            open_price = data['Open'].iloc[0]  # ‘data’資料表中open的第一筆(開盤價)
            # 使用最後一個已知的收盤價，來自前一天 
            previous_low = combined_df_full[f'{prefix}_Low'].iloc[-2] if f'{prefix}_Low' in combined_df_full.columns else None #close改low
            if previous_low is not None: #close改low
                price_change = open_price - previous_low #close改low
                latest_prices[f'{prefix}_price_change'] = price_change 

#===========進行預測======================
print()
print("最新一天的預測結果:")
#取得今日開盤資料後，預測收盤會漲還是跌
if len(latest_prices) == len(tickers) - len(exclude_stocks):  #檢查是否獲得的最新價格變化數據的數量等於所有非黃金股票的數量
    today_df = pd.DataFrame([latest_prices])
    
  #對最新數據進行與訓練數據相同的處理
    today_scaled = scaler.transform(today_df)
    today_classified = pd.DataFrame(today_scaled, columns=feature_columns)
    today_classified = today_classified.applymap(classify_value_X)
    today_encoded = pd.get_dummies(today_classified)
    
  #確保today_encoded的欄位與X_encoded的欄位一致
    today_encoded = today_encoded.reindex(columns=X_encoded.columns, fill_value=0)
    
  prediction = clf.predict(today_encoded)
  print("iShares:", "會漲" if prediction[0] == 1 else "會跌")
  prediction4 = clf4.predict(today_encoded)
  print("Sprott:", "會漲" if prediction4[0] == 1 else "會跌")
  prediction5 = clf5.predict(today_encoded)
  print("SGOL:", "會漲" if prediction5[0] == 1 else "會跌")
else:
    print("無法獲取所有非黃金股票的數據，無法進行預測。")

#特徵重要性計算
feature_importances = clf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X_encoded.columns, 'Importance': feature_importances})

#提取原始特徵名稱
feature_importance_df['Original Feature'] = feature_importance_df['Feature'].apply(lambda x: '_'.join(x.split('_')[:2]))

#按原始特徵名稱分組，求和重要性
feature_importance_summary = feature_importance_df.groupby('Original Feature')['Importance'].sum().reset_index()

#排序並打印特徵重要性
feature_importance_summary.sort_values(by='Importance', ascending=False, inplace=True)
print(feature_importance_summary) 
