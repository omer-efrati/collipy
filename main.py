import logging
import logging.config

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('simpleExample')


# user = input('username: \n')
# password = input('password: \n')




class A:

    def __init__(self, a):
        self.a = a

    @property
    def a(self):
        return self._a, 0.01 * self._a

    @a.setter
    def a(self, value):
        self._a = value
