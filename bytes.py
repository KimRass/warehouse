import torch

tensor = torch.zeros((3, 4), dtype=torch.float32)
# tensor = torch.zeros((3, 4), dtype=torch.float16).cuda()
num_elems = tensor.numel()
# Get size of each element in bytes
element_size = tensor.element_size()
element_size
# Calculate total memory usage in bytes
total_memory_bytes = num_elems * element_size
total_memory_bytes


memory_stats = torch.cuda.memory_stats(torch.cuda.current_device())
allocated_bytes = memory_stats["allocated_bytes.all.current"]
allocated_bytes


a = "hello"
b = ["hello", "python"]
id(a), id(b[0])


x = (1, 2, 3)
y = x
print(id((1, 2, 3)), id((1, 2, 3, 4)), id(x), id(y))
y += (4,)
print(id((1, 2, 3)), id((1, 2, 3, 4)), id(x), id(y))



x = "abc"
y = x
print(id("abc"), id("abcd"), id(x), id(y))
y += "d"
print(id("abc"), id("abcd"), id(x), id(y))


a = [1,2,3]    
b = a          # b와 a는 같은 값을 가르킴(shallow copy)
b is a 

b[1] = 10      # a = b = [1, 10, 3]

a = [5, 11]   # a = [5,11] 이도록 새로 지정함, b와 연결이 끊어짐.
b is a
False
