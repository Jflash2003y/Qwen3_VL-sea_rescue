import json
from collections import Counter

# Windows路径用 r"" 原始字符串，避免反斜杠问题
with open(r'E:\Graduation Project\dataset\seadronessee Objection V2\Compressed Version\Compressed Version\annotations\instances_train.json', 'r') as f:
    data = json.load(f)

# 1. 顶层结构
print('=== 顶层 key ===')
print(list(data.keys()))

# 2. 类别
print('\n=== categories（类别）===')
for cat in data['categories']:
    print(cat)

# 3. 图片数量 + 第一条示例
img_count = len(data['images'])
print(f'\n=== images（共 {img_count} 张）===')
print('第1条：', json.dumps(data['images'][0], indent=2, ensure_ascii=False))

# 4. 标注数量 + 第一条示例
ann_count = len(data['annotations'])
print(f'\n=== annotations（共 {ann_count} 条标注）===')
print('第1条：', json.dumps(data['annotations'][0], indent=2, ensure_ascii=False))

# 5. 各类别标注数量统计
cat_count = Counter(a['category_id'] for a in data['annotations'])
cat_names = {c['id']: c['name'] for c in data['categories']}
print('\n=== 各类别标注数量 ===')
for cid, count in sorted(cat_count.items()):
    name = cat_names.get(cid, 'unknown')
    print(f'  {cid} ({name}): {count} 条')