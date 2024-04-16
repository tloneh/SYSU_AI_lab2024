def ReverseKeyValue(dict1):
    dict2 = {}
    for key, value in dict1.items(): #traverse
        dict2[value] = key #reverse
    return dict2

#test
dict1 = {'Alice':'001', 'Bob':'002'}
print(ReverseKeyValue(dict1))
         