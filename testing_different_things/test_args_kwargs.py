def f(x, y, z):
    return x + y + y ** 2

def try_many_inputs(y, z):
    for i in range(0, 5):
        print(f(i, y, z))

def try_many_inputs2(hehe, *args):
    print(hehe)
    for i in range(0, 5):
        print(f(i, *args))

def try_many_inputs3(hehe, **kwrgs):
    print(hehe)
    for i in range(0, 5):
        print(f(i, **kwrgs))

try_many_inputs(3,4)
try_many_inputs2('hehe', 3,4)

try_many_inputs3('hehe', y=3,z=4, t=3)