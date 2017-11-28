import  os
import  sys
from shutil import copy
#移动指定类型的文本到指定文件夹
a=os.listdir()
os.mkdir('py')
for i in a:
     if os.path.splitext(i)[1]=='.py':
         copy(i,"F:\py\py3.6\\py")
         os.remove(i)