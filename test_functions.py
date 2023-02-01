




class test:
    
    def __init__(self):
        self.a = None
        self.b = None
        self.func = None
        self.T_end = None
        
    def run_simulation(self):
        t=0.0
        while t < self.T_end:
            print(self.func())
            t = t+1.0
            self.b = self.b+1

c_test = test()
c_test.a = 10
c_test.b = 1
c_test.T_end = 3.0

def add():
    return c_test.a+c_test.b

c_test.func = add

c_test.run_simulation()