import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
from datetime import datetime

# 1. 请求网页内容
url = "https://www.factcheck.org/"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:114.0) Gecko/20100101 Firefox/114.0"
}

try:
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
except requests.exceptions.RequestException as e:
    print(f"请求失败: {e}")
    exit()

# 2. 解析HTML
soup = BeautifulSoup(response.text, "html.parser")

# 3. 提取新闻标题和链接
news_list = []
for item in soup.find_all("a", href=True):
    title = item.get_text().strip()
    link = urljoin(url, item["href"])
    
    # 过滤空标题和非正常链接
    if title and link.startswith("http"):
        news_list.append({"title": title, "link": link})

# 4. 保存数据到DataFrame并导出为CSV
news_df = pd.DataFrame(news_list)
news_df.drop_duplicates(inplace=True)
news_df = news_df[news_df["title"].str.len() > 5]

news_df["source"] = "FactCheck"
news_df["date"] = datetime.now().strftime("%Y-%m-%d")
news_df["label"] = "unknown"  # 先标记未知

# 保存为CSV文件
news_df.to_csv("factcheck_scraper.csv", index=False)

print("数据抓取并保存成功！")
