import json

with open(r'E:\Graduation Project\dataset\switch\output\train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f'总样本数: {len(data)}')
print('\n' + '=' * 60)
print('第1条样本:')
print('=' * 60)

sample = data[0]
# 用户提问
user_msg = sample['messages'][0]
for item in user_msg['content']:
    if item['type'] == 'image':
        print(f'图片: {item["image"]}')
    else:
        print(f'提问: {item["text"]}')

# 模型回答
assistant_msg = sample['messages'][1]
print(f'\n回答:\n{assistant_msg["content"][0]["text"]}')

print('\n' + '=' * 60)
print('第2条样本:')
print('=' * 60)

sample2 = data[1]
user_msg2 = sample2['messages'][0]
for item in user_msg2['content']:
    if item['type'] == 'image':
        print(f'图片: {item["image"]}')
    else:
        print(f'提问: {item["text"]}')

assistant_msg2 = sample2['messages'][1]
print(f'\n回答:\n{assistant_msg2["content"][0]["text"]}')