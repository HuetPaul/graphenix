## Graphiques

## Description
Permet de sauvegarder un graphique matplotlib dans un objet dédié pour le réutiliser en l'état plus tard.

Ne fonctionne que pour une seule figure à la fois pour le moment.

## Badges

## Visuals

## Installation

Dans le dossier principal, lancer la commande :

`python setup.py bdist_wheel`

Le fichier wheel d'installation est dans le dossier "dist" nouvellement créé.
Pour installer la librairie dans l'environnement local:

`pip install /path/to/wheelfile.whl`

Ensuite la bibliothèque peut être utilisée comme n'importe quelle autre :
`import graphiques as g`

## Usage

### Pour initialiser un graphique :
- Pour un nouveau graphique vierge: `g=Graphique()`
- Pour un ajouter directement du contenu au graphique (et l'afficher immédiatement):
	- `graph=g.line(x,y,marker="",**args)` équivalent à `plt.plot(x,y,marker="",**args);plt.show()` mais on peut récupérer le graphique
	- `graph=g.errorbar(x,y, err_y, marker="",**args)` équivalent à `plt.errorbar(x,y, err_y, marker="",**args);plt.show()` mais on peut récupérer le graphique

- Pour ouvrir un graphique de nom `n` dans le repertoire `f`
(`n` et `f` deux chaines de caractère, par défaut `f` est le répertoire courant)
ex :
	`f="../Images/Graphiques"` et n=`"test_graphique"`
(l'extension.npz n'est pas obligatoire) :
	- `g=Graphique(n)` si n est dans le repertoire courant
	- `g=Graphique(n,f)` sinon
		
### Pour enregistrer un graphique :
- Attribuer un nom au graphique, sans mettre d'extension :
	`gr.filename="nouveau_nom"`
	Par défaut le nom est graphique_sans_nom. Si on souhaite enregistrer plusieurs graphiques dans le même dossier il est donc important de leur donner un nom (non automatique)
- Éventuellement attribuer un répertoire d'enregistrement :
	`gr.directory="nouveau_repertoire"` Par défaut l'emplacement est celui du repertoire courant
- Pour enregistrer l'objet :
	`gr.save()`
- Pour enregistrer la figure :
	- Attribuer si nécessaire une extension (par défaut l'extension est svg). Les extensions possibles sont celles disponibles via la bibliothèque matplotlib : 'png', 'pdf', 'svg'...

	```
	gr.ext=".nouvelle_extension"
	gr.save_figure()
	```
	
### Pour afficher le graphique :
```gr.show()```
	
### Pour ajouter une ligne de point (equivalent à plt.plot) :
```gr.line(x,y,**args)```
avec `x` et `y` deux listes, tableaux de données à affichées et
`**args` correspond à tous les autres arguments possibles de plot()
Peut être répété autant de fois que nécessaire
	
### Pour ajouter un histogramme :
`g.histogram(valeurs,poids=[],normalisation=True,statistic='sum', bins=10, range=None,**args)` 

où :
- `valeurs` est le tableau, la liste des valeurs à classer
- `poids` est une liste donnant éventuellement un poids pour chaque valeur
- `normalisation` indique si l'histogramme doit être normalisé ou non
- Les autres arguments sont les mêmes que `plt.hist()`

Peut être répété autant de fois que nécessaire
	
### Pour ajouter une image :
`g.image(tableau,axe_x,axe_y,**args)`
où :
- `tableau` représente l'image à afficher
- `axe_x` et `axe_y` donne les graduations des axes de l'image
- `**args` tous les autres arguments possibles pour l'affichage d'une image

### Pour ajouter des contours :
`g.contours(self,contours=np.array([[]]),axe_x=None,axe_y=None,**args) `

- `**args` donnes les arguments possibles pour plt.contours()
- Pour ajouter des lignes de niveau à une image, completer `**args` en laissant les autres arguments par défaut

### Pour ajouter un polygone (surface colorée délimitée par une liste de points) : 
`g.polygon(ind,alpha=0.7, facecolor='C3',**args)`
avec `ind` un tableau/liste de dimension (n,2) où `n` est les nombres de point : `ind[:,0]` correspond aux abscisses des points et `ind[:1]` aux ordonnées

Peut être répété autant de fois que nécessaire

### Pour aller plus loins : affichage de plusieurs graphiques en un :

La structure ne permet pas un enregistrement d'un tel montage, pour cela il faut :
- Initialiser une figure (`plt.figure`...)
- Définir une liste de sous figures (`fig,axs=plt.subplot`...)
chacunes de ces sous figures peut être associée à un graphique `gi` :
	`gi.ax=axi;gi.figure=fig`
- Pour l'affichage il faut appeler pour chaque graphique gi :
`gi.plot()` à la place de `g.show()`
puis appeler les fonctions `plt.show()` ou `plt.save`

## Support

## Roadmap

## Contributing

## Authors and acknowledgment
Paul Huet

## License


## Project status
