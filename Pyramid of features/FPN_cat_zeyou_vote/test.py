import numpy as np
a=np.ones((12,5))
a = a.astype(int)
b=[[0,1,2,3,4],
   [0,0,2,1,3],
   [3,4,1,4,7]]
len=3
m=np.ones(3)
for i in range(len):
    m[i] = np.argmax((np.bincount(b[i])))
    # m[i]=np.argmax((np.apply_along_axis(np.bincount, 1, b[i])))
c=5
print(m)
# print(np.bincount(a))
# a_pre=np.argmax((np.apply_along_axis(np.bincount, 1, a)),axis=1)
# print(a_pre)
# b_pre=np.argmax((np.apply_along_axis(np.bincount, 1, b)),axis=1)
# print(b_pre)
# print(np.argmax((np.apply_along_axis(np.bincount, 1, a)),axis=1))
# a=a[0:c,]
# print(a.size)