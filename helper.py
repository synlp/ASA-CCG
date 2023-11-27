import json


def get_label_list(train_data_path):
    label_list = ['<UNK>']

    with open(train_data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            splits = line.split()
            srl_label = splits[1]
            if srl_label not in label_list:
                label_list.append(srl_label)

    label_list.extend(['[CLS]', '[SEP]'])
    return label_list


def get_tag_label_list(train_data_path):
    label_list = ['<UNK>']

    with open(train_data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            splits = line.split()
            srl_label = splits[2]
            if srl_label not in label_list:
                label_list.append(srl_label)

    label_list.extend(['[CLS]', '[SEP]'])
    return label_list


def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')


def load_json(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        line = f.readline()
    return json.loads(line)
