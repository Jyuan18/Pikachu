# 01
# python 3.6+版本加入了对“类型提示”的支持,用来声明变量的类型
from typing import List, Set, Tuple, Dict


def get_full_name(first_name, last_name):
    fulll_name = first_name.title() + " " + last_name.title()
    return fulll_name


print(get_full_name("john", "doe"))

# John Doe


def get_full_name_2(first_name: str, last_name: str):
    full_name = first_name.title() + " " + last_name.title()
    return full_name


print(get_full_name_2("john", "doe"))

# John Doe


# 02
# 类型检查
def get_name_with_age(name: str, age: int):
    name_with_age = name + " is this old: " + age
    return name_with_age


def get_name_with_age(name: str, age: int):
    name_with_age = name + " is this old: " + str(age)
    return name_with_age


# 03
# 不只是str,你能够声明所有的标准python类型,int\float\bool\bytes等
# 嵌套类型:如有些容器数据结构可以包含其他的值,如dict\list\set\tuple,内部的值可以有自己的类型
# 使用typing标准库来声明这些类型及子类型


def process_items(items: List[str]):  # 表明变量items是一个list,并且该列表中的每一个元素都是str
    for item in items:
        print(item)


# Tuple[int, int, str],变量是一个tuple,前两个元素是int类型,最后一个元素是str类型
# Set[bytes]每个元素是bytes类型
# Dict[str, float],字典需要传入两个子类型,用逗号进行分隔,第一个子类型声明dict所有键,第二个类型声明dict所有值

# 04
# 将类作为类型,将类声明为变量的类型
class Person:
    def __init__(self, name: str):
        self.name = name


def get_person_name(one_person: Person):
    return one_person.name


# 05
# Pydantic是一个用来执行数据校验的python库
# 将数据的结构声明为具有属性的类,每个属性都拥有类型


# 06
# FastAPI中用类型提示做下面几件事
# 编辑器支持\类型检查
# 定义参数要求:声明对请求路径参数\查询参数\请求头\请求体\依赖等要求
# 转换数据\校验数据
# 使用OpenAPI记录API