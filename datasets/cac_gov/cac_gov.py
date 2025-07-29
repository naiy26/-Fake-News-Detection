from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv
import time

def crawl_dynamic_pages():
    # 初始化浏览器
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get("https://wap.cac.gov.cn/hdfw/pypt/gzdt/A09380402phoneindex_1.htm")
    
    with open('output.csv', 'w', newline='', encoding='utf_8_sig') as f:
        writer = csv.writer(f)
        writer.writerow(['标题', '链接', '日期'])
        
        page_num = 1
        max_pages = 5  # 设置最大爬取页数
        
        while page_num <= max_pages:
            print(f"正在处理第 {page_num} 页...")
            
            # 等待内容加载（根据实际页面调整）
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "li.list-item"))
            )
            
            # 解析当前页内容
            items = driver.find_elements(By.CSS_SELECTOR, "li.list-item")
            for item in items:
                try:
                    title = item.find_element(By.TAG_NAME, "a").text
                    link = item.find_element(By.TAG_NAME, "a").get_attribute("href")
                    date = item.find_element(By.CLASS_NAME, "date").text if item.find_elements(By.CLASS_NAME, "date") else ""
                    writer.writerow([title, link, date])
                except Exception as e:
                    print(f"解析条目失败: {e}")
            
            # 尝试点击下一页（根据实际按钮调整）
            try:
                next_btn = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, "//a[contains(text(),'下一页') or contains(text(),'>')]"))
                )
                next_btn.click()
                page_num += 1
                time.sleep(2)  # 等待页面加载
            except:
                print("未找到下一页按钮，可能已到最后一页")
                break
    
    driver.quit()
    print("爬取完成！")

if __name__ == "__main__":
    crawl_dynamic_pages()