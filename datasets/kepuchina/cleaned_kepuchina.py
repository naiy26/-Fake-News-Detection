import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
from datetime import datetime
import re

def clean_text(text):
    """深度清理文本内容"""
    if not text or not isinstance(text, str):
        return ""
    
    # 去除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 替换多种空白字符为单个空格
    text = re.sub(r'\s+', ' ', text)
    # 去除特殊字符，保留中文、英文、数字和常见标点
    text = re.sub(r'[^\w\u4e00-\u9fff\s\-—，。！？、；："\'（）《》]', '', text)
    return text.strip()

def format_date(date_str):
    """严格规范化日期格式"""
    if not date_str or not isinstance(date_str, str):
        return ""
    
    date_str = date_str.strip()
    patterns = [
        (r'(\d{4})年(\d{1,2})月(\d{1,2})日', '%Y-%m-%d'),
        (r'(\d{4})-(\d{1,2})-(\d{1,2})', '%Y-%m-%d'),
        (r'(\d{4})/(\d{1,2})/(\d{1,2})', '%Y/%m/%d'),
        (r'(\d{1,2})月(\d{1,2})日', f'{datetime.now().year}-%m-%d'),
        (r'(\d{4})\.(\d{1,2})\.(\d{1,2})', '%Y.%m.%d')
    ]
    
    for pattern, fmt in patterns:
        match = re.search(pattern, date_str)
        if match:
            try:
                if fmt == '%Y-%m-%d':
                    year = match.group(1)
                    month = match.group(2).zfill(2)
                    day = match.group(3).zfill(2)
                    return f"{year}-{month}-{day}"
                elif fmt == f'{datetime.now().year}-%m-%d':
                    month = match.group(1).zfill(2)
                    day = match.group(2).zfill(2)
                    return f"{datetime.now().year}-{month}-{day}"
                else:
                    date_obj = datetime.strptime(date_str, fmt)
                    return date_obj.strftime('%Y-%m-%d')
            except ValueError:
                continue
    return ""

def validate_url(url):
    """验证并标准化URL"""
    if not url or not isinstance(url, str):
        return ""
    
    url = url.strip()
    if not url.startswith(('http://', 'https://')):
        if url.startswith('/'):
            return f"https://piyao.kepuchina.cn{url}"
        else:
            return f"https://piyao.kepuchina.cn/{url}"
    return url

def get_piyao_rumors():
    base_url = "https://piyao.kepuchina.cn/rumor/rumorlist"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Referer": "https://piyao.kepuchina.cn/",
        "Accept-Language": "zh-CN,zh;q=0.9"
    }
    
    params = {"type": "0", "page": 1}
    all_rumors = []
    max_pages = 5
    
    for page in range(1, max_pages + 1):
        params["page"] = page
        print(f"正在处理第 {page} 页...")
        
        try:
            response = requests.get(base_url, headers=headers, params=params, timeout=15)
            response.encoding = "utf-8"
            
            if response.status_code != 200:
                print(f"请求失败，状态码: {response.status_code}")
                continue
                
            soup = BeautifulSoup(response.text, "html.parser")
            items = soup.select(".rumor-item, .list-item, div[class*=item], li[class*=item]")
            
            for item in items:
                try:
                    # 标题提取与清洗
                    title_elem = item.select_one(".title, .rumor-title, h3, h4, a")
                    title = clean_text(title_elem.get_text()) if title_elem else ""
                    if not title or title.lower() in ["无标题", "暂无标题"]:
                        continue
                    
                    # 链接处理
                    link = validate_url(title_elem.get("href", "")) if title_elem else ""
                    
                    # 日期处理
                    date_elem = item.select_one(".date, .time, .rumor-date, span[class*=date], .pub-time")
                    date_str = clean_text(date_elem.get_text()) if date_elem else ""
                    date = format_date(date_str)
                    
                    # 只保留有效数据
                    if title and (link or date):
                        all_rumors.append({
                            "标题": title,
                            "网站": link,
                            "时间": date
                        })
                    
                except Exception as e:
                    print(f"解析条目出错: {str(e)}")
                    continue
                    
            time.sleep(3)  # 增加延迟避免被封
            
        except Exception as e:
            print(f"第 {page} 页处理失败: {str(e)}")
            continue
    
    return all_rumors

def enhanced_data_cleaning(df):
    """增强的数据清洗流程"""
    # 去除完全重复的数据
    df = df.drop_duplicates(subset=['标题', '网站', '时间'], keep='first')
    
    # 去除标题相似度高的数据（简单版）
    df = df.drop_duplicates(subset=['标题'], keep='first')
    
    # 标准化时间格式
    df['时间'] = pd.to_datetime(df['时间'], errors='coerce').dt.strftime('%Y-%m-%d')
    
    # 标记并处理异常值
    df['标题长度'] = df['标题'].str.len()
    df = df[(df['标题长度'] >= 5) & (df['标题长度'] <= 100)]  # 去除过短或过长的标题
    df = df.drop(columns=['标题长度'])
    
    # 填充空值
    df['网站'] = df['网站'].replace('', pd.NA)
    df['时间'] = df['时间'].replace('', pd.NA)
    
    return df

def save_to_csv(df, filename):
    """保存数据到CSV文件"""
    # 按时间降序排列
    df = df.sort_values(by='时间', ascending=False, na_position='last')
    
    # 重置索引
    df = df.reset_index(drop=True)
    
    # 保存为UTF-8 with BOM编码，确保Excel兼容
    df.to_csv(filename, index=False, encoding='utf_8_sig')
    return df

if __name__ == "__main__":
    print("开始爬取并清洗数据...")
    start_time = time.time()
    
    rumors_data = get_piyao_rumors()
    
    if rumors_data:
        # 转换为DataFrame
        df = pd.DataFrame(rumors_data)
        
        # 执行增强的数据清洗
        df = enhanced_data_cleaning(df)
        
        # 生成输出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"piyao_cleaned_data_{timestamp}.csv"
        
        # 保存数据
        df = save_to_csv(df, filename)
        
        # 打印结果报告
        print(f"\n✅ 数据清洗完成！耗时: {time.time()-start_time:.1f}秒")
        print(f"📊 总记录数: {len(df)}")
        print(f"📁 已保存到: {filename}")
        
        print("\n🔍 数据质量报告:")
        print(f"- 有效标题: {len(df[df['标题'].notna()])}")
        print(f"- 有效链接: {len(df[df['网站'].notna()])}")
        print(f"- 有效时间: {len(df[df['时间'].notna()])}")
        
        print("\n📋 数据预览:")
        print(df.head(10).to_string(index=False))
        
        # 保存质量报告
        report = {
            '总记录数': len(df),
            '有效标题': len(df[df['标题'].notna()]),
            '有效链接': len(df[df['网站'].notna()]),
            '有效时间': len(df[df['时间'].notna()]),
            '最早日期': df['时间'].min(),
            '最近日期': df['时间'].max()
        }
        pd.DataFrame.from_dict(report, orient='index', columns=['统计值']).to_csv(
            f"data_quality_report_{timestamp}.csv", 
            encoding='utf_8_sig'
        )
    else:
        print("❌ 未能获取有效数据，请检查网络或网站结构")