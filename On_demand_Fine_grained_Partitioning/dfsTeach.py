#n代表到第几个数字了,sum代表现在和是多少,m代表第n个数字是什么
def dfs(n, sum, m, nums: list):
    sum += m
    if n==3 and sum != 100: return
    if n==3 and sum == 100:
        print(nums)
        return
    # 列举0-50的数字
    for i in range(0, 50):
        # 加入nums方便最后输出
        nums.append(i)
        dfs(n+1, sum, i, nums)
        # 不用了要把上一个加入的数字删掉
        nums.pop()
dfs(0, 0, 0, [])