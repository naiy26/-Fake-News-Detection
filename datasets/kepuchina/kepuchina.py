import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
from datetime import datetime
import re

def clean_text(text):
    """清理文本中的多余空格和特殊字符"""
    if not text:
        return ""
    # 去除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 替换多种空白字符为单个空格
    text = re.sub(r'\s+', ' ', text)
    # 去除首尾空格
    return text.strip()

def format_date(date_str):
    """规范化日期格式"""
    if not date_str:
        return ""
    
    # 尝试匹配多种日期格式
    patterns = [
        r'(\d{4})年(\d{1,2})月(\d{1,2})日',
        r'(\d{4})-(\d{1,2})-(\d{1,2})',
        r'(\d{4})/(\d{1,2})/(\d{1,2})',
        r'(\d{1,2})月(\d{1,2})日'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, date_str)
        if match:
            if len(match.groups()) == 3:  # 有年份
                year, month, day = match.groups()
            else:  # 只有月和日
                month, day = match.groups()
                year = datetime.now().year
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    
    return date_str  # 无法识别的格式原样返回

def get_piyao_rumors():
    base_url = "https://piyao.kepuchina.cn/rumor/rumorlist"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Referer": "https://piyao.kepuchina.cn/"
    }
    
    params = {"type": "0", "page": 1}
    all_rumors = []
    max_pages = 3  # 减少测试页数
    
    for page in range(1, max_pages + 1):
        params["page"] = page
        print(f"正在爬取第 {page} 页...")
        
        try:
            response = requests.get(base_url, headers=headers, params=params, timeout=15)
            response.encoding = "utf-8"
            
            if response.status_code != 200:
                print(f"请求失败，状态码: {response.status_code}")
                continue
                
            soup = BeautifulSoup(response.text, "html.parser")
            items = soup.select(".rumor-item") or soup.select(".list-item") or soup.find_all("div", class_=lambda x: x and "item" in x)
            
            if not items:
                print(f"第 {page} 页未找到数据，保存HTML供分析...")
                with open(f"debug_page_{page}.html", "w", encoding="utf-8") as f:
                    f.write(response.text)
                continue
                
            for item in items:
                try:
                    # 标题清理
                    title_elem = item.select_one(".title, .rumor-title, h3, h4, a")
                    title = clean_text(title_elem.get_text()) if title_elem else "无标题"
                    
                    # 链接处理
                    link = title_elem.get("href", "") if title_elem else ""
                    if link and not link.startswith("http"):
                        link = f"https://piyao.kepuchina.cn{link}" if link.startswith("/") else f"https://piyao.kepuchina.cn/{link}"
                    
                    # 日期规范化
                    date_elem = item.select_one(".date, .time, .rumor-date, span[class*=date]")
                    date_str = clean_text(date_elem.get_text()) if date_elem else ""
                    date = format_date(date_str)
                    
                    all_rumors.append({
                        "标题": title,
                        "网站": link,
                        "时间": date
                    })
                    
                except Exception as e:
                    print(f"解析条目出错: {str(e)}")
                    continue
                    
            time.sleep(3)  # 增加延迟
            
        except Exception as e:
            print(f"第 {page} 页爬取失败: {str(e)}")
            continue
    
    return all_rumors

def save_to_csv(data, filename):
    """保存数据到CSV并进行额外清理"""
    df = pd.DataFrame(data)
    
    # 确保列顺序
    df = df[["标题", "网站", "时间"]]
    
    # 去除完全空的行
    df.dropna(how="all", inplace=True)
    
    # 保存为CSV
    df.to_csv(filename, index=False, encoding="utf_8_sig")
    print(f"数据已保存到 {filename}")
    
    # 返回清理后的数据
    return df

if __name__ == "__main__":
    print("开始爬取并清理数据...")
    rumors_data = get_piyao_rumors()
    
    if rumors_data:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"piyao_rumors_clean_{timestamp}.csv"
        
        cleaned_df = save_to_csv(rumors_data, filename)
        
        print("\n清理后的数据示例:")
        print(cleaned_df.head(10).to_string(index=False))
        
        print("\n数据统计:")
        print(f"总条数: {len(cleaned_df)}")
        print(f"标题平均长度: {cleaned_df['标题'].str.len().mean():.1f} 字符")
        print(f"最早日期: {cleaned_df['时间'].min()}")
        print(f"最近日期: {cleaned_df['时间'].max()}")
    else:
        print("未能获取有效数据，请检查debug_page_*.html文件分析原因")