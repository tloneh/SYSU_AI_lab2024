def convert_str(sentence):  
#输出子句
        # 如果列表为空，则直接返回空括号  
        if not sentence:  
            return '()'  
        # 如果只有一个单词，则在其后添加逗号  
        if len(sentence) == 1:  
            return '(' + str(sentence[0]) + ',' + ')'  
        return '(' + ','.join(str(word) for word in sentence) + ')'

def ProcessStr(KB_str):
# 将输入字符串转化为元组
        last = 0
        state = 0
        sentences = []
        # 将子句集分割为子句
        for i, item in enumerate(KB_str): 
            if item == '(' :
                if state == 0 :
                    last = i
                state += 1
            if item == ')' :
                state -= 1
                if state == 0 :
                    sentences.append(KB_str[last + 1: i])
        
        # 将子句分割成公式元组
        KB = set()
        for item in sentences:
            sentence = []
            last = 0
            state = 0
            for i, char in enumerate(item):
                if char == '(' :
                    state += 1
                if char == ')' :
                    state -= 1
                    if state == 0 :
                        sentence.append(item[last: i + 1])
                        last = i + 2
                        if last < len(item) and item[last] == '' :
                            last += 1
            KB.add(tuple(sentence))

        return KB
    
def Process_word(word):
#分割一阶公式，得到值，谓词，和项
        value = 1
        if word[0] == '~':
            value = 0
            word = word[1:]
        word = word.replace('(', ',').replace(')', ',')
        splited_word = word.split(',') #最后含有""
        predicate, terms = splited_word[0], splited_word[1:-1] #保留第二个到倒数第二个项
        return predicate, terms, value
    
def unifier(sentence1, sentence2, word1, word2, xigema):  
        index = [""] * 2  # 初始化索引列表  
        # 定义统一句子的函数  
        def unify_sentence(sentence, word, index_idx):  
            if word in sentence and len(sentence) > 1:  
                index[index_idx] = chr(sentence.index(word) + ord('a'))  
            sentence = [w for w in sentence if w != word]  
            for i, word in enumerate(sentence):  
                predicate, terms, value = Process_word(word)  
                terms = [xigema[t] if t in xigema else t for t in terms]  
                sentence[i] = (predicate + '(' + ','.join(terms) + ')' if value else '~' + predicate + '(' + ','.join(terms) + ')')  
            return sentence  
      
        # 对两个句子进行统一处理  
        sentence1 = unify_sentence(list(sentence1), word1, 0)  
        sentence2 = unify_sentence(list(sentence2), word2, 1)  
        # 合并句子并去重  
        new_sentence = list(dict.fromkeys(sentence1 + sentence2))  
      
        return new_sentence, index

def Most_general_unifier(sentence1, sentence2):
        Vari = {'x', 'y', 'z', 'u', 'v', 'w', 'i', 'j', 'k', 'xx', 'yy', 'zz', 'xxx', 'yyy', 'zzz', 'xxxx', 'yyyy', 'zzzz'}
        xigema = {}
        for word1 in sentence1:
            for word2 in sentence2:
                predicate1, terms1, value1 = Process_word(word1)
                predicate2, terms2, value2 = Process_word(word2)
                if predicate1 != predicate2 or value1 == value2:
                    continue
                terms1_cp = terms1[:]
                terms2_cp = terms2[:] #复制一份
                num = len(terms1)
                p = 'Success'
                # 判断是否可以归一
                for i in range(num):
                    term1, term2 = terms1[i], terms2[i]
                    if term1 in Vari and term2 in Vari:
                        terms2 = [term1 if t == term2 else t for t in terms2]
                    elif term1 not in Vari and term2 in Vari:
                        terms2 = [term1 if t == term2 else t for t in terms2]
                    elif term1 in Vari and term2 not in Vari:
                        terms1 = [term2 if t == term1 else t for t in terms1]
                    elif term1 != term2:
                        p = 'Fail'
                        break
                if p == 'Success':
                    terms = terms1 #terms1和terms2相同，改用term更容易理解
                    for i in range(num):
                        term = terms[i]
                        term_old = terms1_cp[i]
                        if term_old != term:
                            xigema[term_old] = term
                            terms1_cp = [term if t == term_old else t for t in terms1_cp]
                            terms2_cp = [term if t == term_old else t for t in terms2_cp]
                    
                        term_old = terms2_cp[i]
                        if term_old != term:
                            xigema[term_old] = term
                            terms1_cp = [term if t == term_old else t for t in terms1_cp]
                            terms2_cp = [term if t == term_old else t for t in terms2_cp]
                    new_sentence, index = unifier(sentence1, sentence2, word1, word2, xigema)
                    return tuple(new_sentence), index, xigema    
        return None, None, None      


def ResolutionFOL(KB_str):
        KB = ProcessStr(KB_str)
        vis = {} # vis字典用于记录已经比较过的句子对，避免重复计算
        dict_idx = {}
        ans = [] #结果
        ans.append("")
        length = 1 #记录子句出现序号
        for sentence in KB:  # 遍历KB中的每个句子，为其分配索引，并添加到结果列表中
            dict_idx[sentence] = length
            ans.append(str(length) + ' ' + convert_str(sentence))
            # print(length, ans[-1])
            length += 1
    
        # 归结推理
        while True :
            quit_1 = False
            for sentence1 in KB:
                quit = 0
                for sentence2 in KB:
                    if sentence1 == sentence2 or (sentence1, sentence2) in vis:
                        continue
                    new_sentence, index, xigema = Most_general_unifier(sentence1, sentence2)
                    if new_sentence != None:
                        vis[(sentence1, sentence2)] = 1
                        if new_sentence in KB: 
                            continue
                        if len(new_sentence) >= max(len(sentence1), len(sentence2)):
                            continue
                        KB.add(new_sentence)
                        dict_idx[new_sentence] = length
                        id = 'R[' + str(dict_idx[sentence1]) + index[0] + ',' + str(dict_idx[sentence2]) + index[1] + ']'
                        trans = []
                        for (key, value) in xigema.items():
                            trans.append(key + '=' + value)
                        trans = '{' + ','.join(trans) + '}' if trans != [] else ""
                        new_sentence = convert_str(new_sentence) if new_sentence != () else '[]'
                        ans.append(str(length) + ' ' + id + trans + ' = ' + new_sentence)
                        length += 1
                        # print(length, ans[-1])
                        quit = 1
                        if new_sentence == '[]':
                            quit_1 = 1
                        break
                if quit == 1: break
            if quit_1 == 1: break
        return ans[1:]
