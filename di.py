# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 09:46:20 2021

@author: Prince
"""

# 将我们的图片写入txt文件

#定义一下文件地址
dir = r'/home/jinfei/PycharmProjects/dataset/400-300-2/train' #写入你自己的数据集所在位置
label = 0

import os
def generate(dir,label):
	files = os.listdir(dir)
	files.sort()
	listText = open('400-300-4_train.txt','a')
	for file in files:
		fileType = os.path.split(file)
		if fileType[1] == '.txt':
			continue
		name = dir+'/' +file + ' ' + str(int(label)) +'\n'
		listText.write(name)
	listText.close()
	
 
outer_path = r'/home/jinfei/PycharmProjects/dataset/400-300-2/train'   #这里是你的图片的目录
 
 
if __name__ == '__main__':
	i = 0
	folderlist = os.listdir(outer_path)          #列举文件夹
	for folder in folderlist:
		generate(os.path.join(outer_path, folder),i)
		i += 1
