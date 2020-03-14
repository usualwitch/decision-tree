s = list('│   │   ├── ')
print(''.join(s))
for i in range(len(s)):
    if s[i] == '│':
        s.insert(i, '*'*2)
print(''.join(s))
