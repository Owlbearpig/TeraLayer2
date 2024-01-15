from enum import Enum


class MyEnum(Enum):
    FIRST = 1
    SECOND = 2
    THIRD = 3


# Print only the attributes (enum members)
for member in MyEnum.__members__:
    print(member)
