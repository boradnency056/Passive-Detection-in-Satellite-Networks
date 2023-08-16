paths = [[(13, 0), (0, 4), (4, 14)], [(13, 0), (0, 5), (5, 14)], [(13, 0), (0, 6), (6, 14)], [(13, 0), (0, 9), (9, 14)], [(13, 0), (0, 10), (10, 14)], [(13, 0), (0, 11), (11, 14)], [(13, 1), (1, 3), (3, 14)], [(13, 1), (1, 4), (4, 14)], [(13, 1), (1, 6), (6, 14)], [(13, 1), (1, 11), (11, 14)], [(13, 2), (2, 3), (3, 14)], [(13, 2), (2, 7), (7, 14)], [(13, 2), (2, 8), (8, 14)], [(13, 2), (2, 12), (12, 14)]]
no_of_satellites = 3

def trim_paths(paths):
	result = []
	for i in range(len(paths)):
		path = paths[i][1]
		result.append(path)
	return result

paths = trim_paths(paths)
print(paths)

obj = {key: 0 for key in range(no_of_satellites)}
# print(obj)
for i in range(len(paths)):
	obj[paths[i][0]] += 1
print(obj)

result = [obj[0]]
for i in range(1, no_of_satellites):
	value = ((result[i-1] * i) + obj[i])/(i+1)
	result.append(int(value))
print(result)
