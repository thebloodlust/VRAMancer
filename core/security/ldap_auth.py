"""
Interopérabilité entreprise :
- Authentification LDAP/Active Directory
- Gestion des utilisateurs et rôles
"""
import ldap3

class LDAPAuthenticator:
    def __init__(self, server_uri, base_dn):
        self.server_uri = server_uri
        self.base_dn = base_dn
        self.server = ldap3.Server(server_uri)

    def authenticate(self, username, password):
        user_dn = f"uid={username},{self.base_dn}"
        try:
            conn = ldap3.Connection(self.server, user=user_dn, password=password, auto_bind=True)
            return True
        except ldap3.LDAPException:
            return False

    def get_user_roles(self, username):
        # À adapter selon le schéma LDAP/AD
        return ["user"]
