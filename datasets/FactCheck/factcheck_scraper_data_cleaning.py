import pandas as pd

# 读取清洗后的CSV文件
df = pd.read_csv("labeled_clean_data.csv")

# 示例1：关键词规则打标签
keywords_fake = ['谣言', '虚假', '假消息', '误导']
df['label'] = df['title'].apply(lambda x: 'fake' if any(keyword in x for keyword in keywords_fake) else 'real')

# 示例2：手动给部分数据打标签
df.loc[10, 'label'] = 'fake'
df.loc[20, 'label'] = 'real'

# 🛠️ 方法2：新增一些手动假新闻数据
new_fake_data = [
    {"title": "震惊！吃大蒜竟能治疗癌症？", "link": "fake_link1", "source": "unknown", "date": "2025-03-26", "label": "fake"},
    {"title": "紧急通知：明天全球互联网关闭一天！", "link": "fake_link2", "source": "unknown", "date": "2025-03-26", "label": "fake"},
    {"title": "科学家发现外星人基地，NASA隐瞒真相", "link": "fake_link3", "source": "unknown", "date": "2025-03-26", "label": "fake"}
]

# 转成DataFrame
fake_df = pd.DataFrame(new_fake_data)

# 拼接到原始数据里
df = pd.concat([df, fake_df], ignore_index=True)

# 检查标签是否增加成功
print(df['label'].value_counts())
print(df.head())

# 保存最终带标签的数据
df.to_csv("labeled_clean_data.csv", index=False)

print("✅ 数据打标签完成，保存成功！")
