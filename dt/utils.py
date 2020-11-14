def is_number(x):
    """
    Return whether string x can be converted into a float.
    """
    try:
        float(x)
    except ValueError:
        return False
    else:
        return True
