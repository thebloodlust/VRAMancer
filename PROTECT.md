# Protection du projet VRAMancer

Pour éviter toute perte ou effacement accidentel :

- **Versionnez tout sur GitHub** (push régulier)
- **Activez la protection de branche** sur la branche principale (main/master)
- **Sauvegardez régulièrement** (cloud, local, export ZIP)
- **N’effacez jamais le dossier .git ni les scripts d’installation**
- **Vérifiez les droits d’accès** avant toute suppression ou modification massive

## Protection de branche GitHub (recommandé)
1. Aller dans Settings > Branches > Add rule
2. Saisir `main` (ou `master`)
3. Cochez :
   - Require a pull request before merging
   - Require status checks to pass before merging
   - Require linear history
   - Include administrators
4. Sauvegardez la règle

## Conseils
- Ne jamais forcer un `git push --force` sur main
- Utilisez des branches pour les développements risqués
- Faites des releases régulières (tags)

---

VRAMancer est un projet critique : protégez-le !
