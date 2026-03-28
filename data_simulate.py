import random

LABELS = ["stress","anxiety","low_mood","academic_pressure","social_strain","fatigue","hopelessness"]

emotion_phrases = {
    "stress": ["压力好大", "事情太多了", "被压得喘不过气"],
    "anxiety": ["有点焦虑", "很紧张", "一直在担心"],
    "low_mood": ["心情很差", "有点难受", "提不起劲"],
    "academic_pressure": ["作业好多", "论文写不完", "考试要来了"],
    "social_strain": ["不太想社交", "和朋友有点尴尬", "关系变僵了"],
    "fatigue": ["好累", "很疲惫", "一点精神都没有"],
    "hopelessness": ["感觉没意义", "有点绝望", "不知道为什么要这样"]
}

connectors = ["，", "，真的", "，而且", "，甚至", "，感觉", "，有时候"]

endings = ["", "啊", "……", "真的", "唉"]

def generate_sample():
    num_labels = random.randint(2,4)  # 多标签
    selected = random.sample(LABELS, num_labels)

    text_parts = []
    label_vector = [0]*len(LABELS)

    for label in selected:
        phrase = random.choice(emotion_phrases[label])
        text_parts.append(phrase)
        label_vector[LABELS.index(label)] = 1

    # 拼接
    sentence = ""
    for i, part in enumerate(text_parts):
        if i == 0:
            sentence += part
        else:
            sentence += random.choice(connectors) + part

    sentence += random.choice(endings)

    return {"text": sentence, "labels": label_vector}


# 生成200条
data = [generate_sample() for _ in range(200)]

# 看一下
for i in range(100):
    print(data[i])