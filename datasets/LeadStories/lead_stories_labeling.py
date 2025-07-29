import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# 初始化 VADER 分析器
analyzer = SentimentIntensityAnalyzer()

# 读取清洗后的 CSV 数据
cleaned_data_path = r"C:\Users\User\Documents\挑战杯比赛\LeadStories\cleaned_lead_stories_facts.csv"
output_path = "lead_stories_labeled.csv"

# 定义辅助关键词，用于加强分类逻辑
positive_keywords = ["confirmed", "true", "real", "verified", "authentic", "genuine"]
negative_keywords = ["hoax", "fake", "false", "debunked", "misleading", "rumor"]

# 定义情感分类函数（结合 TextBlob 和 VADER）
def classify_label(title):
    # TextBlob 情感分析
    textblob_sentiment = TextBlob(title).sentiment.polarity

    # VADER 情感分析（适合短文本）
    vader_sentiment = analyzer.polarity_scores(title)['compound']

    # 情感阈值
    threshold = 0.05

    # 如果 TextBlob 和 VADER 均为正情感，则归为 True
    if textblob_sentiment > threshold or vader_sentiment > threshold:
        return "True"
    # 如果 TextBlob 和 VADER 均为负情感，或者出现负面关键词，则归为 False
    elif textblob_sentiment < -threshold or vader_sentiment < -threshold or \
            any(keyword in title.lower() for keyword in negative_keywords):
        return "False"
    # 如果存在明显正面关键词则归为 True
    elif any(keyword in title.lower() for keyword in positive_keywords):
        return "True"
    # 如果情感值在阈值范围内，则为 Unknown
    else:
        return "Unknown"

try:
    # 读取数据文件
    df = pd.read_csv(cleaned_data_path)
    
    # 应用分类函数，处理每个标题
    df['label'] = df['title'].apply(classify_label)
    
    # 保存带标签的数据到新的 CSV 文件
    df.to_csv(output_path, index=False)
    
    print("✅ 数据打标签完成，已保存到 lead_stories_labeled.csv")

except Exception as e:
    print(f"❌ 出错了: {e}")
