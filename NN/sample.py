from collections import deque

a = deque(maxlen = 5)


a.append((1,2,3,4,5))
a.append((2,3,4,5,6))
a.append((3,4,5,6,7,))
a.append((4,5,6,7,8,))
a.append((5,6,7,8,9,))
print(a)

a.append((6,7,8,9,0))
print(a)