import pandas as pd

# 读取原始数据
df = pd.read_csv(r"LeadStories/lead_stories_facts.csv")

# 1️⃣ 去除空值和无效数据
df.dropna(subset=["title", "link"], inplace=True)

# 2️⃣ 去重
df.drop_duplicates(subset=["title", "link"], inplace=True)

# 3️⃣ 规范日期格式
# 转换为统一的 YYYY-MM-DD 格式
df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

# 4️⃣ 处理特殊字符
df["title"] = df["title"].str.replace(r"\n", " ", regex=True).str.strip()
df["title"] = df["title"].str.replace(r"\s+", " ", regex=True)  # 清除多余空格

# 5️⃣ 补全标签
df["label"].fillna("unverified", inplace=True)

# 保存清洗后的数据
df.to_csv("cleaned_lead_stories_facts.csv", index=False)

print("🎉 数据清洗完成，已保存到 'cleaned_lead_stories_facts.csv'")
