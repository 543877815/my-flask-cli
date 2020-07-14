class Scope:
    allow_api = []
    allow_module = []
    forbidden = []

    # 权限链式相加
    def __add__(self, other):
        self.allow_api = self.allow_api + other.allow_api
        self.allow_api = list(set(self.allow_api))

        self.allow_module = self.allow_module + other.allow_module
        self.allow_module = list(set(self.allow_module))

        self.forbidden = self.forbidden + other.allow_module
        self.forbidden = list(set(self.forbidden))


class AdminScope(Scope):
    allow_api = ['v1.user+super_get_user',
                 'v1.user+super_delete_user']

    # allow_module = ['v1.user']

    def __init__(self):
        self + UserScope()
        print(self.allow_api)


class UserScope(Scope):
    forbidden = ['v1.user+super_get_user',
                 'v1.user+super_delete_user']

    allow_api = ['v1.A', 'v1.B']


class SuperScope(Scope):
    allow_api = ['v1.C', 'v1.D']
    allow_module = ['v1.user']

    def __init__(self):
        self + UserScope()


def is_in_scope(scope, endpoint):
    # scope()
    # 反射
    # v1.view_func   v1.module_name+view_func
    # v1.red_name+view_func
    scope = globals()[scope]()
    splits = endpoint.split('+')
    red_name = splits[0]
    if endpoint in scope.forbidden:
        return False
    if endpoint in scope.allow_api:
        return True
    if red_name in scope.allow_module:
        return True
    else:
        return False
