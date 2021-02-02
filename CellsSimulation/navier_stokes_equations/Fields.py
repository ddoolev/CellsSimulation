class Fields2:

    def __init__(self, field_v, field_u, field_p):
        self._v = field_v
        self._u = field_u
        self._p = field_p

    def copy(self):
        return Fields2(self._v, self._u, self._p)

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, field_v):
        self._v = field_v

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, field_u):
        self._u = field_u

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, field_p):
        self._p = field_p


