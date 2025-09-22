# Ce fichier rend le sous-dossier 'network' importable comme un package Python.
# Tu peux exposer ici les classes et fonctions principales pour simplifier les imports.

from .transport import Transport
from .interface_selector import select_best_interface, list_interfaces
