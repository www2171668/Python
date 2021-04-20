# -- encoding:utf-8 --
"""
Create by ibf on 2018/8/26
"""


class A:
    def ff(self):
        self.f = 10
        print("F:{}".format(self.f))

    def gf(self):
        self.f = self.f + 10
        print("G:{}".format(self.f))

    def hf(self):
        self.f -= 2
        print("G:{}".format(self.f))


a = A()
a.ff()
a.gf()
a.hf()
a.gf()
print(a.f)
