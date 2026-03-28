import random
import pandas as pd

N = 1000  # 生成数量

# ===== 模板库 =====
emotions = ["开心", "难过", "焦虑", "生气", "平静"]
scenes = ["工作", "学习", "生活", "社交"]

moments_templates = [
    "今天{scene}真的有点{emotion}",
    "{scene}一天，感觉{emotion}",
    "最近状态：{emotion}",
    "生活就是这样，有点{emotion}",
]

comments_templates = [
    "哈哈你这也太{emotion}了",
    "别想太多啦",
    "加油，会好起来的",
    "出来吃饭放松一下",
]

chat_templates = [
    "A: 在吗\nB: 在，怎么了\nA: 最近有点{emotion}",
    "A: 今天{scene}好累\nB: 辛苦了\nA: 有点{emotion}",
]

status_templates = [
    "状态：{emotion}",
    "状态：今天{emotion}",
    "状态：{scene}中，{emotion}",
]

# ===== 标签（S2） =====
label_map = {
    "开心": 2,
    "平静": 1,
    "难过": 0,
    "焦虑": 0,
    "生气": 0,
}

# ===== 生成函数 =====
def gen_text():
    scene = random.choice(scenes)
    emotion = random.choice(emotions)

    category = random.choice(["moments", "comments", "chat", "status"])

    if category == "moments":
        text = random.choice(moments_templates)
    elif category == "comments":
        text = random.choice(comments_templates)
    elif category == "chat":
        text = random.choice(chat_templates)
    else:
        text = random.choice(status_templates)

    text = text.format(scene=scene, emotion=emotion)

    label = label_map.get(emotion, 1)
    return text, label

# ===== 生成数据 =====
data = []
for _ in range(N):
    text, label = gen_text()
    data.append({
        "text": text,
        "label": label
    })

df = pd.DataFrame(data)
df.to_csv("wechat_style_data.csv", index=False, encoding="utf-8-sig")

print("生成完成，共{}条".format(len(df)))