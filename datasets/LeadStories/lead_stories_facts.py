from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time

# 设置ChromeDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

try:
    # 打开网页
    url = "https://leadstories.com/"
    driver.get(url)
    print("成功打开网页！")

    # 等待页面加载
    time.sleep(5)  # 视页面加载速度适当调整等待时间

    # 获取网页内容
    page_source = driver.page_source

    # 关闭浏览器
    driver.quit()

    # 用BeautifulSoup解析网页
    soup = BeautifulSoup(page_source, "html.parser")

    # 提取文章内容
    articles = []
    for item in soup.find_all("article"):  # 假设每个事实检查信息在<article>标签中
        title = item.find("h2")  # 假设标题在<h2>标签中
        link = item.find("a")["href"] if item.find("a") else ""
        description = item.find("p").get_text(strip=True) if item.find("p") else ""
        # 你可以根据实际的 HTML 结构修改这些元素的提取方法

        # 添加 source, date, label 信息
        source = "Lead Stories"  # 来源固定为 Lead Stories
        date = item.find("time").get_text(strip=True) if item.find("time") else "Unknown"  # 假设时间在<time>标签中
        label = "unknown"  # label 可能根据实际网站内容添加，你可以根据具体内容调整

        articles.append({
            "title": title.get_text(strip=True) if title else "无标题",
            "link": link,
            "source": source,
            "date": date,
            "label": label
        })

    # 保存数据到DataFrame
    df = pd.DataFrame(articles)
    df.to_csv("lead_stories_facts.csv", index=False)

    print("数据已保存到 lead_stories_facts.csv")

except Exception as e:
    print(f"出错了: {e}")
    driver.quit()
