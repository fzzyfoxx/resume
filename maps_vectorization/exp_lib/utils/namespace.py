
def no_var_error(var_name, message=None):
    if var_name in locals():
        return locals()['var_name']
    else:
        if message is not None:
            print(message)
        raise NameError(f'Variable {var_name} not found')
