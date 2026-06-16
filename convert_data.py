import json
import os

def convert_format(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    labels_set = set()
    result = []
    
    for idx, item in enumerate(data):
        text = item['text']
        entities = item.get('entities', [])
        
        labels = []
        for j, ent in enumerate(entities):
            start = ent['start_idx']
            end = ent['end_idx']
            entity_type = ent['type'].upper()
            entity_text = ent['entity']
            labels.append([f"T{j}", entity_type, start, end, entity_text])
            labels_set.add(entity_type)
        
        result.append({
            "id": idx,
            "text": text,
            "labels": labels
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)
    
    return labels_set

train_labels = convert_format('data/CMeEE/mid_data/CMeEE-train.json', 'data/CMeEE/mid_data/train.json')
dev_labels = convert_format('data/CMeEE/mid_data/CMeEE-dev.json', 'data/CMeEE/mid_data/dev.json')

all_labels = list(train_labels.union(dev_labels))
print(f"实体类型: {all_labels}")
print(f"类型数量: {len(all_labels)}")

#with open('data/CMeEE/mid_data/labels.json', 'w', encoding='utf-8') as f:
#    json.dump(all_labels, f, ensure_ascii=False)


print("数据转换完成！")
