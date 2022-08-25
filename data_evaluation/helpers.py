
def is_iterable(obj):
    try:
        iter(obj)
    except TypeError:
        return False
    else:
        return True

if __name__ == '__main__':
    print(is_iterable(3.4))
    print(is_iterable([3.4]))