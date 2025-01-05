x = 5
y = 8
z = 0

for a in range(1, 10 + 1):
  z = 0
  for num in range(a-2):
    z = z + num ** 3 + x * num + num - y

  if z == 120:
    print(a)
