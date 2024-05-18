import random

# 加载数据集
def load_dataset(filename):
    dataset = []
    with open(filename, 'r') as file:
        for line in file.readlines():
            row = line.strip().split(',')
            dataset.append(row)
    return dataset

# 数据预处理
def preprocess_dataset(dataset):
    X = []  # 特征
    y = []  # 标签
    for row in dataset[1:]:  # 忽略数据集的首行
        # 将特征值转换为数值类型
        features = [1 if row[i] == 'YES' else 0 if row[i] == 'NO' else 0 if row[i] == 'Single' else 
                    1 if row[i] == 'Divorced' else 2 if row[i] == 'Married' else int(row[i]) for i in range(4)]
        X.append(features)  # 前四列为特征值
        # 标签为1表示“有风险”，0表示“好”
        label = 1 if int(row[2]) <= 30000 else 0  # 第三列为TaxableIncome
        y.append(label)  # 添加到标签列表中
    return X, y

# 划分数据集为训练集和测试集
def train_test_split(dataset, split_ratio=0.3, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)
    train_size = int(len(dataset) * split_ratio)
    train_set = []
    test_set = list(dataset)
    while len(train_set) < train_size:
        index = random.randrange(len(test_set))
        train_set.append(test_set.pop(index))
    return train_set, test_set

# 构建决策树模型
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

def test_split(index, value, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def gini_index(groups, classes):
    n_instances = float(sum(len(group) for group in groups))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances)
    return gini

def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    # 检查是否没有分割
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # 检查最大深度
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # 左节点
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    # 右节点
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)

# 使用决策树进行预测
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# 计算准确率
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 1.0

# 主函数
def main():
    # 加载数据集
    dataset = load_dataset('DT_data.csv')
    # 数据预处理
    X, y = preprocess_dataset(dataset)
    
    split_ratio = 0.3
    seed = 86
    # 划分数据集为训练集和测试集
    train_set, test_set = train_test_split(list(zip(X, y)), split_ratio=split_ratio, random_seed=seed)

    # 构建决策树模型
    max_depth = 5
    min_size = 10
    tree = build_tree(train_set, max_depth, min_size)

    # 在测试集上进行预测
    actual = [row[-1] for row in test_set]
    predicted = [predict(tree, row[:-1]) for row in test_set]

    # 计算准确率
    accuracy = accuracy_metric(actual, predicted)
    if accuracy>0.9:
        # print(f"split_ratio:{seed} Accuracy:{accuracy}")
        print(f"Accuracy:{accuracy} with split_ratio:{split_ratio} and seed:{seed}")

# 执行主函数
main()