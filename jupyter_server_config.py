from jupyter_server.auth import IdentityProvider

# Token Authentication (Default Behavior)
c.ServerApp.token = 'raglab'
# c.ServerApp.password = ''
# c.IdentityProvider.token = ''
# c.PasswordIdentityProvider.hashed_password = ''
c.ServerApp.disable_check_xsrf = False

# Network settings
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True
