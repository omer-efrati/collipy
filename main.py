from collider import Collider
import logging
import logging.config

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('simpleExample')

user = input('username: \n')
password = input('password: \n')


x = Collider(user, password)
