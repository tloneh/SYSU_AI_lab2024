class StuData:
    def __init__(self, file_):
        # 初始化data
        self.data = [] 

        try:
            #打开文件
            with open(file_, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    info = line.strip().split()
                    if len(info) == 4 :
                        name, stu_num, gender, age_ = info
                        age = int(age_)
                        #用字典存储，方便后续分类排序
                        self.data.append({'name': name, 'stu_num': stu_num, 'gender': gender, 'age': age})

        except FileNotFoundError:
            print(f"Can't find the file: {file_}")
    
    def AddData(self, name, stu_num, gender, age):
        self.data.append({'name': name, 'stu_num': stu_num, 'gender': gender, 'age': age})

    def SortData(self, key):
        try:
            self.data.sort(key = lambda x: x[key])
        except KeyError:
            print(f"The key: {key} does not exist.")

    def ExportFile(self, out_file):
        try:
            with open(out_file, 'w') as f:
                for i in self.data:
                    f.write(f"{i['name']} {i['stu_num']} {i['gender']} {i['age']}\n")
        except IOError:
            print(f'Error to open {out_file}. Reason: {IOError}')
