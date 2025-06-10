def append_or_clear(left, right):
    if right=='__clear__':
        return []
    elif right is None:
        return left
    elif left is None:
        return [right]
    else:
        return left + [right]
            
def extend_or_clear(left, right):
    if right=='__clear__':
        return []
    elif right is None:
        return left
    elif left is None:
        return right
    else:
        return left + right