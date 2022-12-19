import os

def check_charset(file_path):
    import chardet
    with open(file_path, "rb") as f:
        data = f.read(4)
        charset = chardet.detect(data)['encoding']
    return charset
'./Bn0_paddle.txt'
'./Bn0_torch.txt'


f1 = open('./tinynet_torch.txt', 'r',encoding=check_charset('./tinynet_torch.txt'))
f1 = f1.readlines()
f1 = [i.rstrip() for i in f1]
# print(f1)
f2 = open('./f.txt','r',encoding=check_charset('./f.txt'))
f2 = f2.readlines()
f2 = [i.rstrip() for i in f2]
# a = {}
# a[f1.readlines()] = f2.readlines()
# print(a)

# coding=utf-8

d = dict(zip(f1, f2))
print(d)



# your_path = 你的文件路径
# with open(your_path, encoding=check_charset(your_path)) as f:
#     data = f.read()
#     print(data)