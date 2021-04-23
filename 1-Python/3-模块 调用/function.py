
def words(filename,number):
    print("create %s a file here %d."%(filename,number))

def Shuzi():
    print("1")

if __name__ == '__main__':
    words('yiwen')
    Shuzi()


class Restaurant:
    def __init__(self,restaurant_name):
        self.restaurant_name = restaurant_name
    def describe_restaurant(self, dish):
        print('酒店名称是%s，菜品是%s'%(self.restaurant_name, dish))

if __name__ == "__main__":
    R1 = Restaurant('大酒店')
    R1.describe_restaurant('辣鸡')

