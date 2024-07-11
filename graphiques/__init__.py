# -*- coding: utf-8 -*-
"""
l'objectif de cette bibliothèque est de faciliter la création et la manipulation de graphiques.
Elle permet en particulier de sauvegarder le graphique en tant que tel,
améliorant ainsi la répétabilité.
Elle contient un objet Graphique dédié à cette manipulation et des méthodes
générant des graphiques de bases sur le modèle des méthodes de matplotlib :
  - plt.plot, plt.loglog, plt.semilogx ...
"""
from __future__ import annotations

import numpy as np
from matplotlib import axes
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.colors import LogNorm
from matplotlib.colors import to_rgba
from matplotlib.colors import to_hex
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as sp
from scipy.interpolate import griddata

c: str = "ysqgsdkbn"  # chaine de caractère permettant de combler d'éventuels
# vides dans les listes (à ne pas utiliser dans les manipulations de graphiques

# Définitions couleurs :
C1: str = '#6307ba'  # Violet
C2: str = '#16b5fa'  # Cyan
C3: str = '#2ad500'  # Vert clair
C4: str = '#145507'  # vert foncé
C5: str = '#ff8e00'  # Orange
C6: str = '#cb0d17'  # Rouge
C7: str = '#5694b2'  # Bleu pastel
C8: str = '#569a57'  # Vert pastel
C9: str = '#b986b9'  # Lavande
C10: str = '#c6403c'  # Rouge pastel
C11: str = '#d39d5d'  # Beige
C12: str = '#25355d'  # Bleu ?
C13: str = '#fcc100'  # Jaune
C14: str = '#7ab5fa'  # Bleu ciel
C15: str = '#fc2700'  # Orange foncé
C16: str = '#0fc88f'  # Bleu-Vert
C17: str = '#a8173b'  # Rouge cerise
C18: str = '#1812c4'  # Bleu foncé
C19: str = "#000000"  # Noir
C20: str = "#707070"  # Gris

l_couleurs: list[str] = [C1, C2, C3, C4, C5, C6, C7, C8, C9,
                         C10, C11, C12, C13, C14, C15, C16, C17, C18, C19, C20]


def linear_color_interpolation(val: np.float_ | float | list[np.float_] | list[float] | np.ndarray,
                               val_min: np.float_ = 0., val_max: np.float_ = 1.,
                               col_min: str | tuple = C1, col_max: str = C2
                               ) -> str | list[str] | np.ndarray:
    """
    Return a color/list of colors which is linearly interpolated between the two extremal colors col_min
    and col_max
    :param col_min: color associated with the minimal value (in hexadecimal or rgba)
    :param col_max: color associated with the maximal value (in hexadecimal or rgba)
    :param val: the value to be interpolated
    :param val_min: the minimal value
    :param val_max: the maximal value
    :return: the interpolated color(s) in hex
    """
    if type(col_min) is str:
        col_min: tuple = to_rgba(col_min)
    if type(col_max) is str:
        col_max: tuple = to_rgba(col_max)
    col_min: np.ndarray = np.array(col_min)
    col_max: np.ndarray = np.array(col_max)

    if type(val) is np.float_ or type(val) is float:
        return to_hex(tuple(col_min + (col_max - col_min) * (val - val_min) / (val_max - val_min)))
    else:
        res: list[str] = []
        for v in val:
            res.append(to_hex(tuple(col_min + (col_max - col_min) * (v - val_min) / (val_max - val_min))))

    if type(val) is np.ndarray:
        return np.array(res)
    elif type(val) is list[np.float_] or type(val) is list[float]:
        return res
    else:
        raise UserWarning("The values to be interpolated has the wrong type :", type(val),
                          "the only accepted types are float, np.float_, list[np.float_| float], np.ndarray")


def fusion_liste(L: list, nombre: bool = True) -> np.ndarray:
    """Convertit une liste de listes de tailles différentes en un tableau 2d de numpy
    les "trous" sont remplis avec des np.nan"""
    if len(L) == 0:
        return np.array(L)
    lmax: int = 0
    for li in L:
        lmax = max(lmax, len(li))
    if lmax == 0:
        return np.array(L)
    res: list = []
    for i in range(len(L)):
        li: list = list(L[i])
        if len(li) < lmax:
            for j in range(max(0, len(li) - 1), lmax - 1):
                if nombre:
                    li.append(np.nan)
                else:
                    li.append(c)
        res.append(li)
    return np.array(res)


def récupération_liste(L: np.ndarray, nombre: bool = True) -> list:
    """Récupère la liste de liste originale"""
    res: list = []
    for i in range(len(L)):
        liste: list = []
        j: int = 0
        while j < len(L[0]) and not (nombre and np.isnan(L[i, j])) and L[i, j] != c:
            # ~ print(lum[i,j])
            liste.append(L[i, j])
            j += 1
        # ~ print(lum)
        res.append(np.array(liste))
    return res


# noinspection PyTypeChecker
class Graphique:
    """
l'objectif de cet objet est de faciliter la creation et la manipulation de graphiques
Il contient toutes la variables necessaries pour tracer un Graphique sur un axe ax
Il peut lancer l'affichage du Graphique, son enregistrement en .png et peut se sauvegarder en utilisant un format
.npy afin d'éviter de tout recalculer en permanence

#------------------------Méthodes de manipulation-------------------------------------------------------------------

Pour initialiser un Graphique :
    - Pour un nouveau Graphique : g=Graphique()
    - Pour ouvrir un Graphique de nom n dans le repertoire f (n et f deux chaines de caractère,
     par défaut f est le répertoire courant) ex: f="../Images/Graphiques" et n="test_graphique"
     (l'extension.npz n'est pas obligatoire) :
        g=Graphique(n) si n est dans le repertoire courant
        g=Graphique(n,f) sinon
Pour enregistrer un Graphique :
    - Attribuer un nom au Graphique, sans mettre d'extension :
        g.nom_fichier="nouveau_nom"
        Par défaut le nom est graphique_sans_nom. Si on souhaite enregistrer plusieurs Graphique dans le même dossier
         il est donc important de leurs donner un nom (non automatique)
    - Éventuellement attribuer un répertoire d'enregistrement :
        g.emplacement="nouveau_repertoire" Par défaut l'emplacement est celui du repertoire courant
    - Pour enregistrer l'objet :
        g.enregistrement()
    - Pour enregistrer la figure :
        - Attribuer si nécessaire une extension (par défaut l'extension est svg).
         Les extension possibles sont celles disponible via la bibliothèque matplotlib : 'png', 'pdf', 'svg'...
            g.ext=".nouvelle_extension"
        g.enregistrement_figure()
Pour afficher le Graphique :
    g.affichage_figure()
Pour ajouter une ligne de point (equivalent à plt.plot) :
    g.ajout_ligne(x,y,**args) avec x et y deux listes, tableaux de données à afficher
     et **args correspond à touts les autres argument possibles de plot()
    Peut être répété autant de fois que nécessaire
Pour ajouter un histogramme :
    g.,ajout_histogramme(valeurs,poids=[],normalisation=True,statistic='sum', bins=10, range=None,**args) :
    où - valeurs est la tableau, la liste des valeurs à classer
       - poids est une liste donnant éventuellement un poids pour chaque valeurs
       - normalisation indique si l'histogramme doit être normalisé ou non
       - Les autres arguments sont les mêmes que plt.hist()
    Peut être répété autant de fois que nécessaire
Pour ajouter une image :
    g.ajout_image(tableau,axe_x,axe_y,**args)
    où  tableau représente l'image à afficher
        axe_x et axe_y donne les graduations des axes de l'image
        **args tous les autres argument possibles pour l'affichage d'une image
Pour ajouter des contours :
    ajout_contours(self,contours=np.array([[]]),axe_x=None,axe_y=None,**args) :
    - **args donnes les arguments possibles pour plt.contours()
    - Pour ajouter des lignes de niveau à une images completer **args en laissant les autres arguments par défaut
Pour ajouter un polygone (surface colorée délimitée par une liste de points) : 
    g.ajout_polygone(ind,alpha0=0.7, facecolor='C3',**args)
    avec ind un tableau/liste de dimension (n,2) où n est les nombre de point : ind[:,0] correspond aux abscisses des
    points et et ind[:1] aux ordonnées
    Peut être répété autant de fois que nécessaire

Pour aller plus loin : affichage de plusieurs graphiques n un :
    La structure ne permet pas un enregistrement d'un tel montage, pour cela il faut :
    initialiser une figure (plt.figure..)
    définir une liste de sous figures (fig,axs=plt.subplot(...)
    chacunes de ces sous figures peut être associée à un Graphique gi :
        gi.ax=axi;gi.figure=fig
    Pour l'affichage il faut appeler pour chaque Graphique gi :
        gi.projection_figure() à la place de g.affichage_figure()
    Pour l'affichage ou l'enregistrement appeler les fonctions plt.show() ou plt.save(...)


    """

    def __init__(self, nom_fichier: str = "", emplacement: str = ""):
        self.nom_fichier: str = nom_fichier
        if nom_fichier == "":
            self.emplacement: str = "./"
            # s.repertoire+"/graphique_sans_nom"
            self.nom_fichier: str = "graphique_sans_nom"
        self.ext: str = ".svg"
        self.titre: str = ""
        self.style: str = 'default'  # style global :
        # styles disponible : plt.style.available : 'default' 'Solarize_Light2'
        # '_classic_test_patch' '_mpl-gallery' '_mpl-gallery-nogrid'
        # 'bmh' 'classic' 'dark_background' 'fast' 'fivethirtyeight' 'ggplot'
        # 'grayscale' 'seaborn' 'seaborn-bright' 'seaborn-colorblind' 'seaborn-dark' 'seaborn-dark-palette'
        # 'seaborn-darkgrid' 'seaborn-deep' 'seaborn-muted' 'seaborn-notebook' 'seaborn-paper' 'seaborn-pastel'
        # 'seaborn-poster' 'seaborn-talk' 'seaborn-ticks' 'seaborn-white' 'seaborn-whitegrid' 'tableau-colorblind10'
        self.param_police: dict = {
            "font.size": 13}  # contient des paramètres additionnels pour gérer globalement la police
        self.ax: axes = None
        # contient des paramètres additionnels pour ax ex : xlabel='x',xscale='log'...
        self.param_ax: dict = dict()
        self.barre_couleurs: Colorbar = None
        self.param_colorbar: dict = dict()  # contient des paramètres
        # additionnels pour la colorbar ex : label="legende"...
        self.param_bords: dict = dict(fontsize=15, inline=True)
        # contient des paramètres additionnels pour l'affichage des bords
        self.couleurs_bords: list[str] = []
        self.fig: Figure = None
        # contient des paramètres additionnels pour la figure ex : facecolor="w"...
        self.param_fig: dict = dict()
        # contient des paramètres additionnels pour
        self.param_enrg_fig: dict = dict(bbox_inches="tight")
        # l'enregistrement de la figure ex : dpi=300...
        # liste des coordonnées des repères de l'axe x
        self.axe_x: np.ndarray[float] = np.array([-np.inf])
        self.noms_axe_x: np.ndarray[str] = np.array(
            ["vide"])  # liste des noms des repères de l'axe x
        # liste des coordonnées des repères de l'axe y
        self.axe_y: np.ndarray[float] = np.array([-np.inf])
        self.noms_axe_y: np.ndarray[str] = np.array(
            ["vide"])  # liste des noms des repères de l'axe y
        self.lignes_x: list[
            list[float | np.float_]] = []  # liste contenant des listes de coordonnées x à afficher via plt.plot
        self.lignes_y: list[
            list[float | np.float_]] = []  # liste contenant des listes de coordonnées x à afficher via plt.plot
        self.lignes_t_x: list[
            list[float | np.float_]] = []  # liste contenant des listes de coordonnées x à afficher via plt.text
        self.lignes_t_y: list[
            list[float | np.float_]] = []  # liste contenant des listes de coordonnées x à afficher via plt.text
        self.lignes_t_s: list[
            list[float | np.float_]] = []  # liste contenant des listes de chaines de caractères à afficher via plt.text
        self.err_y: list[list[
            float | np.float_]] = []  # liste contenant des listes des erreurs associées aux coordonnées y
        # à afficher via plt.plot
        # contient les dictionnaires de paramètres complémentaires
        self.param_lignes: list = []
        # contient les dictionnaires de paramètres complémentaires
        self.param_texts: list = []
        # pour plt.plot(self.lignes_x[i],self.lignes_y[i]
        # liste contenant des listes de coordonnées des bords des barres des
        self.bords_histogramme: list = []
        # histogrammes
        # liste contenant les valeurs des hauteurs de barres des histogrammes
        self.vals_histogramme: list = []
        # contient les dictionnaires de paramètres complémentaires pour plt.bar
        self.param_histogrammes: list = []
        self.param_légende: dict = dict()  # paramètres de plt.légende
        # tableau à afficher via pcolor, les limites sont axe_x/y
        self.tableau: np.ndarray = np.array([[]])
        self.axe_x_tableau: np.ndarray = np.array(
            [])  # liste des coordonnées x de tableau
        self.axe_y_tableau: np.ndarray = np.array(
            [])  # liste des coordonnées y de tableau
        # paramètres de pcolor colorbar,legend...
        self.param_tableau: dict = dict()
        self.contours: np.ndarray = np.array(
            [[]])  # tableau à afficher via contour, les limites sont axe_x/y décalées d'un demi bin
        self.clabels: np.ndarray = np.array(
            [])  # Éventuels str à afficher pour chaques niveaux de contours
        self.clabels_mask: np.ndarray = np.array(
            [])  # Masque où liste d'indices des labels à afficher pour les contours
        # paramètres de contours, alpha0,label...
        self.param_contours: dict = dict()
        self.axe_x_contours: np.ndarray = np.array(
            [])  # liste des coordonnées x de contours
        self.axe_y_contours: np.ndarray = np.array(
            [])  # liste des coordonnées y de contours
        # listes des niveaux de contours à afficher
        self.niveaux: np.ndarray = np.array([])
        # nombre de contours à afficher, ignore si niveaux != np.array([])
        self.nb_contours: int = 10
        # utilise tableau par défaut pour tracer les contours
        self.contours_est_tableau: bool = False
        # donne les indices de polygones à afficher en surbrillance
        self.indices_polygones: list = []
        self.param_polygones: list = []
        self.grille: bool = False  # lance la commande plt.grid()
        if nom_fichier != "":
            valeurs: np.lib.npyio.NpzFile = np.load(nom_fichier + ".npz")
            self.nom_fichier = nom_fichier
            if emplacement != "":
                self.emplacement = emplacement
            elif "/" in emplacement:
                i: int = emplacement.find("/")
                while emplacement.find("/") > 0:
                    # Cherche la dernière occurrence de "/"
                    i = emplacement.find("/")
                self.emplacement = nom_fichier[:i]
                self.nom_fichier = nom_fichier[i:]
            else:
                self.emplacement = "./"
            if "ext" in valeurs:
                self.ext = str(valeurs["ext"])
            self.ax = None
            self.param_ax = dict()
            if 'style' in valeurs:
                self.style = str(valeurs["style"])
            if "param_colorbar" in valeurs:
                if len(valeurs["nom_param_colorbar"].shape) != 1:
                    print("erreur lors de la lecture des paramètres de la colorbar")
                # print(valeurs["param_colorbar"],valeurs["nom_param_colorbar"])
                for i in range(len(valeurs["nom_param_colorbar"])):
                    self.param_colorbar[valeurs["nom_param_colorbar"]
                                        [i]] = valeurs["param_colorbar"][i]
                    try:
                        self.param_colorbar[valeurs["nom_param_colorbar"][i]] = float(
                            valeurs["param_colorbar"][i])
                    except ValueError:
                        try:
                            self.param_colorbar[valeurs["nom_param_colorbar"][i]] = str(
                                valeurs["param_colorbar"][i])
                        except ValueError:
                            self.param_colorbar[valeurs["nom_param_colorbar"]
                                                [i]] = valeurs["param_colorbar"][i]
                    # print(self.param_colorbar[valeurs["nom_param_colorbar"][i]]," n est pas un reel")
                # else :
                # print("Le paramètre de ",valeurs["nom_param_colorbar"]," : ",
                # self.param_colorbar[valeurs["nom_param_colorbar"][i]]," a ete transforme en reel")

            if "param_bords" in valeurs:
                if len(valeurs["nom_param_bords"].shape) != 1:
                    print("erreur lors de la lecture des paramètres de la colorbar")
                # print(valeurs["param_bords"],valeurs["nom_param_bords"])
                for i in range(len(valeurs["nom_param_bords"])):
                    self.param_bords[valeurs["nom_param_bords"]
                                     [i]] = valeurs["param_bords"][i]
                    try:
                        self.param_bords[valeurs["nom_param_bords"][i]] = float(
                            valeurs["param_bords"][i])
                    except ValueError:
                        try:
                            self.param_bords[valeurs["nom_param_bords"][i]] = str(
                                valeurs["param_bords"][i])
                        except ValueError:
                            self.param_bords[valeurs["nom_param_bords"]
                                             [i]] = valeurs["param_bords"][i]
                    # print(self.param_bords[valeurs["nom_param_bords"][i]]," n est pas un reel")
                # else :
                # print("Le paramètre de ",valeurs["nom_param_bords"]," : ",
                # self.param_bords[valeurs["nom_param_bords"][i]]," a ete transforme en reel")
            if "couleurs_bords" in valeurs:
                self.couleurs_bords = valeurs["couleurs_bords"]
            if "param_police" in valeurs:
                if len(valeurs["nom_param_police"].shape) != 1:
                    print("erreur lors de la lecture des paramètres de police")
                # print(valeurs["param_police"],valeurs["nom_param_police"])
                for i in range(len(valeurs["nom_param_police"])):
                    self.param_police[valeurs["nom_param_police"]
                                      [i]] = valeurs["param_police"][i]
                    try:
                        self.param_police[valeurs["nom_param_police"][i]] = float(
                            valeurs["param_police"][i])
                    except ValueError:
                        try:
                            self.param_police[valeurs["nom_param_police"][i]] = str(
                                valeurs["param_police"][i])
                        except ValueError:
                            self.param_police[valeurs["nom_param_police"]
                                              [i]] = valeurs["param_police"][i]
                    # print(self.param_police[valeurs["nom_param_police"][i]]," n est pas un reel")
                # else :
                # print("Le paramètre de ",valeurs["nom_param_police"]," : ",
                # self.param_police[valeurs["nom_param_police"][i]]," a ete transforme en reel")
            if "param_ax" in valeurs:
                if len(valeurs["nom_param_ax"].shape) != 1:
                    print("erreur lors de la lecture des paramètres de ax")
                # ~ print(valeurs["param_ax"],valeurs["nom_param_ax"])
                for i in range(len(valeurs["nom_param_ax"])):
                    self.param_ax[valeurs["nom_param_ax"]
                                  [i]] = valeurs["param_ax"][i]
                    try:
                        self.param_ax[valeurs["nom_param_ax"]
                                      [i]] = float(valeurs["param_ax"][i])
                    except ValueError:
                        try:
                            self.param_ax[valeurs["nom_param_ax"]
                                          [i]] = str(valeurs["param_ax"][i])
                        except ValueError:
                            self.param_ax[valeurs["nom_param_ax"]
                                          [i]] = valeurs["param_ax"][i]
                    # print(self.param_ax[valeurs["nom_param_ax"][i]]," n est pas un reel")
                # else :
                # print("Le paramètre de ",valeurs["nom_param_ax"]," : ",
                # self.param_ax[valeurs["nom_param_ax"][i]]," a ete transforme en reel")
            self.barre_couleurs = None
            if "param_fig" in valeurs:
                if len(valeurs["nom_param_fig"].shape) != 1:
                    print("erreur lors de la lecture des paramètres de la figure")
                # ~ print(valeurs["param_ax"],valeurs["nom_param_ax"])
                for i in range(len(valeurs["nom_param_fig"])):
                    self.param_fig[valeurs["nom_param_fig"]
                                   [i]] = valeurs["param_fig"][i]
                    try:
                        self.param_fig[valeurs["nom_param_fig"]
                                       [i]] = float(valeurs["param_fig"][i])
                    except ValueError:
                        try:
                            self.param_fig[valeurs["nom_param_fig"][i]] = str(
                                valeurs["param_fig"][i])
                        except ValueError:
                            self.param_fig[valeurs["nom_param_fig"]
                                           [i]] = valeurs["param_fig"][i]
            if "param_enrg_fig" in valeurs:
                if len(valeurs["nom_param_enrg_fig"].shape) != 1:
                    print(
                        "erreur lors de la lecture des paramètres pour l'enregistrement de la figure")
                # ~ print(valeurs["param_ax"],valeurs["nom_param_ax"])
                for i in range(len(valeurs["nom_param_enrg_fig"])):
                    self.param_enrg_fig[valeurs["nom_param_enrg_fig"]
                                        [i]] = valeurs["param_enrg_fig"][i]
                    try:
                        self.param_enrg_fig[valeurs["nom_param_enrg_fig"][i]] = float(
                            valeurs["param_enrg_fig"][i])
                    except ValueError:
                        try:
                            self.param_enrg_fig[valeurs["nom_param_enrg_fig"][i]] = str(
                                valeurs["param_enrg_fig"][i])
                        except ValueError:
                            self.param_enrg_fig[valeurs["nom_param_enrg_fig"]
                                                [i]] = valeurs["param_enrg_fig"][i]
            if "axe_x" in valeurs:
                # liste des coordonnées des repères de l'axe x
                self.axe_x = valeurs["axe_x"]
            if "noms_axe_x" in valeurs:
                self.noms_axe_x = valeurs["noms_axe_x"]
            if "axe_y" in valeurs:
                # liste des coordonnées des repères de l'axe y
                self.axe_y = valeurs["axe_y"]
            if "noms_axe_y" in valeurs:
                self.noms_axe_y = valeurs["noms_axe_y"]
            if "lignes_x" in valeurs:
                self.lignes_x = récupération_liste(valeurs["lignes_x"],
                                                   nombre=True)
                # liste contenant des listes de coordonnées x à afficher via plt.plot
            if "lignes_y" in valeurs:
                self.lignes_y = récupération_liste(valeurs["lignes_y"],
                                                   nombre=True)
                # liste contenant des listes de coordonnées y à afficher via plt.
            if "lignes_t_x" in valeurs:
                self.lignes_t_x = récupération_liste(
                    valeurs["lignes_t_x"], nombre=True)
                # liste contenant des listes de coordonnées x à afficher via plt.text
            if "lignes_t_y" in valeurs:
                self.lignes_t_y = récupération_liste(
                    valeurs["lignes_t_y"], nombre=True)
                # liste contenant des listes de coordonnées y à afficher via plt.text
            if "lignes_t_s" in valeurs:
                self.lignes_t_s = récupération_liste(
                    valeurs["lignes_t_s"], nombre=False)
                # liste contenant des listes de chaine de caractère à afficher via plt.text
            if "err_y" in valeurs:
                self.err_y = récupération_liste(valeurs["err_y"], nombre=True)
                # liste contenant des listes des erreurs associées aux coordonnées y à afficher via plt.errorbar
            self.param_lignes = []
            if "param_lignes" in valeurs:
                param_lignes = récupération_liste(
                    valeurs["param_lignes"], nombre=False)
                noms_param_lignes = récupération_liste(
                    valeurs["nom_param_lignes"], nombre=False)
                for i in range(len(param_lignes)):
                    self.param_lignes.append(dict())
                    for j in range(len(param_lignes[i])):
                        self.param_lignes[i][noms_param_lignes[i]
                                             [j]] = param_lignes[i][j]
                        try:
                            self.param_lignes[i][noms_param_lignes[i][j]] = float(
                                param_lignes[i][j])
                        except ValueError:
                            try:
                                self.param_lignes[i][noms_param_lignes[i][j]] = str(
                                    param_lignes[i][j])
                            except ValueError:
                                self.param_lignes[i][noms_param_lignes[i]
                                                     [j]] = param_lignes[i][j]

                            # print(param_lignes[i][j] ," n est pas un reel")
                    # else :
                    # print("Le paramètre de ",noms_param_lignes[i][j]," : ",param_lignes[i][j] ,
                    # " a ete transforme en reel")
            self.param_texts = []
            if "param_texts" in valeurs:
                param_texts = récupération_liste(
                    valeurs["param_texts"], nombre=False)
                noms_param_texts = récupération_liste(
                    valeurs["nom_param_texts"], nombre=False)
                for i in range(len(param_texts)):
                    self.param_texts.append(dict())
                    for j in range(len(param_texts[i])):
                        self.param_texts[i][noms_param_texts[i]
                                            [j]] = param_texts[i][j]
                        try:
                            self.param_texts[i][noms_param_texts[i][j]] = float(
                                param_texts[i][j])
                        except ValueError:
                            try:
                                self.param_texts[i][noms_param_texts[i][j]] = str(
                                    param_texts[i][j])
                            except ValueError:
                                self.param_texts[i][noms_param_texts[i]
                                                    [j]] = param_texts[i][j]

                            # print(param_texts[i][j] ," n est pas un reel")
                    # else :
                    # print("Le paramètre de ",noms_param_texts[i][j]," : ",param_texts[i][j] ,
                    # " a ete transforme en reel")
            if "bords_histogramme" in valeurs:
                self.bords_histogramme = récupération_liste(
                    valeurs["bords_histogramme"], nombre=True)
            if "vals_histogramme" in valeurs:
                self.vals_histogramme = récupération_liste(valeurs["vals_histogramme"],
                                                           nombre=True)
                # liste contenant des listes de coordonnées x à afficher via plt.plot
            self.param_histogrammes = []
            if "param_histogrammes" in valeurs:
                param_histogrammes: list[np.ndarray] = récupération_liste(
                    valeurs["param_histogrammes"], nombre=False)
                noms_param_histogrammes: list[np.ndarray[str]] = récupération_liste(valeurs["nom_param_histogrammes"],
                                                                                    nombre=False)
                for i in range(len(param_histogrammes)):
                    self.param_histogrammes.append(dict())
                    for j in range(len(param_histogrammes[i])):
                        self.param_histogrammes[i][noms_param_histogrammes[i]
                                                   [j]] = param_histogrammes[i][j]
                        try:
                            self.param_histogrammes[i][noms_param_histogrammes[i][j]] = float(
                                param_histogrammes[i][j])
                        except ValueError:
                            try:
                                self.param_histogrammes[i][noms_param_histogrammes[i][j]] = str(
                                    param_histogrammes[i][j])
                            except ValueError:
                                self.param_histogrammes[i][noms_param_histogrammes[i]
                                                           [j]] = param_histogrammes[i][j]
                    # else :
                    # print("Le paramètre de ",noms_param_histogrammes[i][j]," : ",param_histogrammes[i][j] ,
                    # " a ete transforme en reel")
            if "param_légende" in valeurs:
                for i in range(len(valeurs["nom_param_légende"])):
                    self.param_légende[valeurs["nom_param_légende"]
                                       [i]] = valeurs["param_légende"][i]
                    try:
                        self.param_légende[valeurs["nom_param_légende"][i]] = float(
                            valeurs["param_légende"][i])
                    except ValueError:
                        try:
                            self.param_légende[valeurs["nom_param_légende"][i]] = str(
                                valeurs["param_légende"][i])
                        except ValueError:
                            self.param_légende[valeurs["nom_param_légende"]
                                               [i]] = valeurs["param_légende"][i]
                    # print(self.param_légende[valeurs["nom_param_légende"][i]]," n est pas un reel")
                # else :
                # print("Le paramètre de ", valeurs["nom_param_légende"], " : ",
                # self.param_légende[valeurs["nom_param_légende"][i]]," a ete transforme en reel")
            if "tableau" in valeurs:
                # tableau a afficher via pcolor, les limites sont axe_x/y
                self.tableau = valeurs["tableau"]
                self.param_tableau = dict()
                if len(valeurs["param_tableau"].shape) != 1:
                    print("erreur lors de la lecture des paramètres du tableau")
                # ~ print(valeurs["param_tableau"],valeurs["nom_param_tableau"])
                for i in range(len(valeurs["param_tableau"])):
                    self.param_tableau[valeurs["nom_param_tableau"]
                                       [i]] = valeurs["param_tableau"][i]
                    try:
                        self.param_tableau[valeurs["nom_param_tableau"][i]] = float(
                            valeurs["param_tableau"][i])
                    except ValueError:
                        try:
                            self.param_tableau[valeurs["nom_param_tableau"][i]] = str(
                                valeurs["param_tableau"][i])
                        except ValueError:
                            self.param_tableau[valeurs["nom_param_tableau"]
                                               [i]] = valeurs["param_tableau"][i]
                    # print(self.param_tableau[valeurs["nom_param_tableau"][i]]," n est pas un reel")
                # else :
                # print("Le paramètre de ",valeurs["nom_param_tableau"][i]," : ",valeurs["param_tableau"][i],
                # " a ete transforme en reel")
                # liste des coordonnées x de contours
                self.axe_x_tableau = valeurs["axe_x_tableau"]
                # liste des coordonnées y de contours
                self.axe_y_tableau = valeurs["axe_y_tableau"]
            if "contours" in valeurs:
                self.contours = valeurs[
                    "contours"]  # tableau à afficher via contour, les limites sont axe_x/y décalées d'un demi bin
                # liste des coordonnées x de contours
                self.axe_x_contours = valeurs["axe_x_contours"]
                # liste des coordonnées y de contours
                self.axe_y_contours = valeurs["axe_y_contours"]
            if "param_contours" in valeurs:
                self.param_contours = dict()
                if len(valeurs["param_contours"].shape) != 1:
                    print("erreur lors de la lecture des paramètres des contours")
                # ~ print(valeurs["param_contours"],valeurs["nom_param_contours"])
                for i in range(len(valeurs["param_contours"])):
                    self.param_contours[valeurs["nom_param_contours"]
                                        [i]] = valeurs["param_contours"][i]
                    try:
                        self.param_contours[valeurs["nom_param_contours"][i]] = float(
                            valeurs["param_contours"][i])
                    except ValueError:
                        try:
                            self.param_contours[valeurs["nom_param_contours"][i]] = str(
                                valeurs["param_contours"][i])
                        except ValueError:
                            self.param_contours[valeurs["nom_param_contours"]
                                                [i]] = valeurs["param_contours"][i]
                        # print(valeurs["param_contours"][i]," n est pas un reel")
                # else :
                # print("Le paramètre de ",valeurs["nom_param_contours"][i]," : ",
                # valeurs["param_contours"][i]," a ete transforme en reel")
            if "niveaux" in valeurs:
                # listes des niveaux de contours à afficher
                self.niveaux = valeurs["niveaux"]
                if "clabels" in valeurs:
                    self.clabels = valeurs["clabels"]
                if "clabels_mask" in valeurs:
                    self.clabels_mask = valeurs["clabels_mask"]
            if "paramètres" in valeurs:
                self.titre = valeurs["paramètres"][0]
                self.nb_contours = int(valeurs["paramètres"][1])
                self.contours_est_tableau = bool(int(valeurs["paramètres"][2]))
                self.grille = bool(int(valeurs["paramètres"][3]))
            if "indices_polygones" in valeurs:
                ind1 = récupération_liste(
                    valeurs["indices_polygones_1"], nombre=True)
                ind2 = récupération_liste(
                    valeurs["indices_polygones_2"], nombre=True)
                for i in range(len(ind1)):
                    self.indices_polygones.append(
                        np.array([ind1[i], ind2[i]]).T)
                param_polygones = récupération_liste(
                    valeurs["param_polygones"], nombre=False)
                nom_param_polygones = récupération_liste(
                    valeurs["nom_param_polygones"], nombre=False)
                for i in range(len(param_polygones)):
                    self.param_polygones.append(dict())
                    for j in range(len(param_polygones[i])):
                        self.param_polygones[i][nom_param_polygones[i]
                                                [j]] = param_polygones[i][j]
                        try:
                            self.param_polygones[i][nom_param_polygones[i][j]] = float(
                                param_polygones[i][j])
                        except ValueError:
                            try:
                                self.param_polygones[i][nom_param_polygones[i][j]] = str(
                                    param_polygones[i][j])
                            except ValueError:
                                self.param_polygones[i][nom_param_polygones[i]
                                                        [j]] = param_polygones[i][j]
                        # print(param_polygones[i][j] ," n est pas un reel")
                    # else :
                    # print("Le paramètre de ",nom_param_polygones[i][j]," : ",param_polygones[i][j] ,
                    # " a ete transforme en reel")
            valeurs.close()

    def enregistrement(self) -> None:
        """Enregistre le Graphique dans un fichier .npz
        :rtype: None
        """
        enrg: dict = dict()  # dictionnaire contenant toutes las variables à enregistrer, argument
        # de np.savez_compressed(nom_fichier,**enrg)
        enrg["ext"] = self.ext
        enrg["style"] = self.style
        if len(self.param_colorbar) > 0:
            param_colorbar: list = []
            nom_param_colorbar: list[str] = []
            for clef in self.param_colorbar.keys():
                param_colorbar.append(self.param_colorbar[clef])
                nom_param_colorbar.append(clef)
            enrg["param_colorbar"] = np.array(param_colorbar)
            enrg["nom_param_colorbar"] = np.array(nom_param_colorbar)
        if len(self.param_bords) > 0:
            param_bords: list = []
            nom_param_bords: list[str] = []
            for clef in self.param_bords.keys():
                param_bords.append(self.param_bords[clef])
                nom_param_bords.append(clef)
            enrg["param_bords"] = np.array(param_bords)
            enrg["nom_param_bords"] = np.array(nom_param_bords)
        if len(self.param_police) > 0:
            param_police: list = []
            nom_param_police: list[str] = []
            for clef in self.param_police.keys():
                param_police.append(self.param_police[clef])
                nom_param_police.append(clef)
            enrg["param_police"] = np.array(param_police)
            enrg["nom_param_police"] = np.array(nom_param_police)
        if len(self.param_ax) > 0:
            param_ax: list = []
            nom_param_ax: list[str] = []
            for clef in self.param_ax.keys():
                param_ax.append(self.param_ax[clef])
                nom_param_ax.append(clef)
            enrg["param_ax"] = np.array(param_ax)
            enrg["nom_param_ax"] = np.array(nom_param_ax)
        if len(self.param_fig) > 0:
            param_fig: list = []
            nom_param_fig: list[str] = []
            for clef in self.param_fig.keys():
                param_fig.append(self.param_fig[clef])
                nom_param_fig.append(clef)
            enrg["param_fig"] = np.array(param_fig)
            enrg["nom_param_fig"] = np.array(nom_param_fig)
        if len(self.param_enrg_fig) > 0:
            param_enrg_fig: list = []
            nom_param_enrg_fig: list[str] = []
            for clef in self.param_enrg_fig.keys():
                param_enrg_fig.append(self.param_enrg_fig[clef])
                nom_param_enrg_fig.append(clef)
            enrg["param_enrg_fig"] = np.array(param_enrg_fig)
            enrg["nom_param_enrg_fig"] = np.array(nom_param_enrg_fig)
        if len(self.axe_x) == 0 or self.axe_x[0] > -np.inf:
            enrg["axe_x"] = self.axe_x
        if len(self.noms_axe_x) == 0 or self.noms_axe_x[0] != "vide":
            enrg["noms_axe_x"] = self.noms_axe_x
        if len(self.axe_y) == 0 or self.axe_y[0] > -np.inf:
            enrg["axe_y"] = self.axe_y
        if len(self.noms_axe_y) == 0 or self.noms_axe_y[0] != "vide":
            enrg["noms_axe_y"] = self.noms_axe_y
        if len(self.lignes_x) > 0:
            enrg["lignes_x"] = fusion_liste(self.lignes_x, nombre=True)
        if len(self.lignes_y) > 0:
            enrg["lignes_y"] = fusion_liste(self.lignes_y, nombre=True)
        if len(self.lignes_t_x) > 0:
            enrg["lignes_t_x"] = fusion_liste(self.lignes_t_x, nombre=True)
        if len(self.lignes_t_y) > 0:
            enrg["lignes_t_y"] = fusion_liste(self.lignes_t_y, nombre=True)
        if len(self.lignes_t_s) > 0:
            enrg["lignes_t_s"] = fusion_liste(self.lignes_t_s)
        if len(self.err_y) > 0:
            enrg["err_y"] = fusion_liste(self.err_y, nombre=True)
        if len(self.param_lignes) > 0:
            param_lignes: list = []
            nom_param_lignes: list[list[str]] = []
            for i in range(len(self.param_lignes)):
                param_lignes.append([]), nom_param_lignes.append([])
                for clef in self.param_lignes[i].keys():
                    param_lignes[i].append(self.param_lignes[i][clef])
                    nom_param_lignes[i].append(clef)
            enrg["param_lignes"] = fusion_liste(param_lignes, nombre=False)
            enrg["nom_param_lignes"] = fusion_liste(
                nom_param_lignes, nombre=False)
        if len(self.param_texts) > 0:
            param_texts: list = []
            nom_param_texts: list[list[str]] = []
            for i in range(len(self.param_texts)):
                param_texts.append([]), nom_param_texts.append([])
                for clef in self.param_texts[i].keys():
                    param_texts[i].append(self.param_texts[i][clef])
                    nom_param_texts[i].append(clef)
            enrg["param_texts"] = fusion_liste(param_texts, nombre=False)
            enrg["nom_param_texts"] = fusion_liste(
                nom_param_texts, nombre=False)
        if len(self.bords_histogramme) > 0:
            enrg["bords_histogramme"] = fusion_liste(
                self.bords_histogramme, nombre=True)
        if len(self.vals_histogramme) > 0:
            enrg["vals_histogramme"] = fusion_liste(
                self.vals_histogramme, nombre=True)
        if len(self.param_histogrammes) > 0:
            param_histogrammes: list = []
            nom_param_histogrammes: list[list[str]] = []
            for i in range(len(self.param_histogrammes)):
                param_histogrammes.append([])
                nom_param_histogrammes.append([])
                for clef in self.param_histogrammes[i].keys():
                    param_histogrammes[i].append(
                        self.param_histogrammes[i][clef])
                    nom_param_histogrammes[i].append(clef)
            enrg["param_histogrammes"] = fusion_liste(
                param_histogrammes, nombre=False)
            enrg["nom_param_histogrammes"] = fusion_liste(
                nom_param_histogrammes, nombre=False)
        if len(self.param_légende) > 0:
            param_légende: list = []
            nom_param_légende: list[str] = []
            for clef in self.param_légende.keys():
                param_légende.append(self.param_légende[clef])
                nom_param_légende.append(clef)
            enrg["param_légende"] = np.array(param_légende)
            enrg["nom_param_légende"] = np.array(nom_param_légende)
        if len(self.tableau) > 1:
            enrg["tableau"] = self.tableau
            enrg["axe_x_tableau"] = self.axe_x_tableau
            enrg["axe_y_tableau"] = self.axe_y_tableau
            param_tableau: list = []
            nom_param_tableau: list[str] = []
            for clef in self.param_tableau.keys():
                param_tableau.append(self.param_tableau[clef])
                nom_param_tableau.append(clef)
            print(param_tableau, nom_param_tableau)
            enrg["param_tableau"] = np.array(param_tableau)
            enrg["nom_param_tableau"] = np.array(nom_param_tableau)
        if len(self.contours) > 1:
            enrg["contours"] = self.contours
            enrg["axe_x_contours"] = self.axe_x_contours
            enrg["axe_y_contours"] = self.axe_y_contours
        if len(self.couleurs_bords) > 0:
            enrg["couleurs_bords"] = self.couleurs_bords
        if len(self.param_contours) > 0:
            param_contours: list = []
            nom_param_contours: list[str] = []
            for clef in self.param_contours.keys():
                param_contours.append(self.param_contours[clef])
                nom_param_contours.append(clef)
            enrg["param_contours"] = np.array(param_contours)
            enrg["nom_param_contours"] = np.array(nom_param_contours)
        if len(self.niveaux) > 0:
            enrg["niveaux"] = self.niveaux
            if len(self.clabels) > 0:
                enrg["clabels"] = self.clabels
            if len(self.clabels_mask) > 0:
                enrg["clabels_mask"] = self.clabels_mask
        param = [self.titre, str(self.nb_contours), str(
            int(self.contours_est_tableau)), str(int(self.grille))]
        enrg["paramètres"] = param
        if len(self.indices_polygones) > 0:
            ind1: list[float | np.double] = []
            ind2: list[float | np.double] = []
            for liste in self.indices_polygones:
                ind1.append(liste.T[0])
            for liste in self.indices_polygones:
                ind2.append(liste.T[1])
            enrg["indices_polygones_1"] = fusion_liste(ind1, nombre=True)
            enrg["indices_polygones_2"] = fusion_liste(ind2, nombre=True)
            param_polygones: list = []
            nom_param_polygones: list[list[str]] = []
            for i in range(len(self.param_polygones)):
                param_polygones.append([])
                nom_param_polygones.append([])
                for clef in self.param_polygones[i].keys():
                    param_polygones[i].append(self.param_polygones[i][clef])
                    nom_param_polygones[i].append(clef)
            enrg["param_polygones"] = fusion_liste(
                param_polygones, nombre=False)
            enrg["nom_param_polygones"] = fusion_liste(
                nom_param_polygones, nombre=False)
        np.savez_compressed(self.emplacement + "/" +
                            self.nom_fichier + ".npz", **enrg)

    def ligne(self, x, y=None, marker: str = "", **args) -> None:
        """Équivalent plt.plot"""
        if type(y) is str:
            marker = y
            y = None
        if y is None:
            y = np.copy(x)
            x = np.arange(0, len(y))
        if type(x) is list:
            x = np.array(x)
        if type(y) is list:
            y = np.array(y)

        if (len(x.shape) == 2 and len(y.shape) == 2 and
                x.shape[1] == y.shape[1]):
            for (X, Y, i) in zip(x, y, np.arange(len(x))):
                self.lignes_x.append(X)
                self.lignes_y.append(Y)
                self.err_y.append([])
                args_auxi: dict = {}
                for k in args.keys():
                    if (isinstance(args[k], list) | isinstance(args[k], np.ndarray)) and len(args[k]) == len(y):
                        args_auxi[k] = args[k][i]
                    else:
                        args_auxi[k] = args[k]
                if marker != "" and not ("linestyle" in args_auxi):
                    args_auxi["linestyle"] = ""
                    if ((isinstance(marker, list) | isinstance(marker, np.ndarray))
                            and len(marker) == len(y)):
                        args_auxi["marker"] = marker[i]
                    else:
                        args_auxi["marker"] = marker
                elif marker != "":
                    if ((isinstance(marker, list) | isinstance(marker, np.ndarray))
                            and len(marker) == len(y)):
                        args_auxi["marker"] = marker[i]
                    else:
                        args_auxi["marker"] = marker
                if "color" not in args:
                    args_auxi["color"] = l_couleurs[i % len(l_couleurs)]
                if "label" in args and args_auxi["label"] == "":
                    del args_auxi["label"]
                    # supprime les légendes vides
                    # (évite l'affichage d'un message d'erreur et d'un cadre gris vide à la place de la légende)
                self.param_lignes.append(args_auxi)
        elif len(y.shape) == 2 and y.shape[1] == len(x):
            for (Y, i) in zip(y, np.arange(len(y))):
                self.lignes_x.append(x)
                self.lignes_y.append(Y)
                self.err_y.append([])
                args_auxi: dict = {}
                for k in args.keys():
                    if (isinstance(args[k], list) | isinstance(args[k], np.ndarray)) and len(args[k]) == len(y):
                        args_auxi[k] = args[k][i]
                    else:
                        args_auxi[k] = args[k]
                if marker != "" and not ("linestyle" in args_auxi):
                    args_auxi["linestyle"] = ""
                    if ((isinstance(marker, list) | isinstance(marker, np.ndarray))
                            and len(marker) == len(y)):
                        args_auxi["marker"] = marker[i]
                    else:
                        args_auxi["marker"] = marker
                elif marker != "":
                    if ((isinstance(marker, list) | isinstance(marker, np.ndarray))
                            and len(marker) == len(y)):
                        args_auxi["marker"] = marker[i]
                    else:
                        args_auxi["marker"] = marker
                if "color" not in args:
                    args_auxi["color"] = l_couleurs[i % len(l_couleurs)]
                if "label" in args and args_auxi["label"] == "":
                    del args_auxi["label"]
                    # supprime les légendes vides
                    # (évite l'affichage d'un message d'erreur et d'un cadre gris vide à la place de la légende)
                self.param_lignes.append(args_auxi)
        elif len(y) != len(x):
            raise (ValueError(
                "Attention : pour l'ajout d'une ligne dans un Graphique, la liste des ordonnées doit être "
                "de la même taille que la liste des abscisses : x : "
                + str(x.shape) + " y : " + str(y.shape)))
        else:
            self.lignes_x.append(x)
            self.lignes_y.append(y)
            self.err_y.append([])
            if marker != "" and not ("linestyle" in args):
                args["linestyle"] = ""
                args["marker"] = marker
            elif marker != "":
                args["marker"] = marker
            if "color" not in args:
                # args["color"] = 'C' + str(len(self.lignes_x) % 10)
                args["color"] = l_couleurs[(
                    len(self.lignes_x) - 1) % len(l_couleurs)]
            if "label" in args and args["label"] == "":
                del args["label"]
                # supprime les légendes vides (évite l'affichage d'un message d'erreur et d'un cadre gris
                # vide à la place de la légende)
            self.param_lignes.append(args)

    def text(self, x, y, s, **args) -> None:
        """Équivalent plt.text"""
        if type(s) is str:
            s: np.ndarray = np.array([s for X in x])
        if (len(np.shape(np.array(x))) == 2 and len(np.shape(np.array(y))) == 2 and
                np.shape(np.array(x))[1] == np.shape(np.array(y))[1]):
            for (X, Y, S) in zip(x, y, s):
                self.lignes_t_x.append(X)
                self.lignes_t_y.append(Y)
                self.lignes_t_s.append(S)
                args_auxi: list[dict] = [{} for XX in X]
                for k in args.keys():
                    if type(args[k]) is list | np.ndarray and len(args[k]) == len(X):
                        for i in range(len(X)):
                            args_auxi[i][k] = args[k][i]
                    else:
                        for i in range(len(X)):
                            args_auxi[i][k] = args[k]
                if "color" not in args:
                    for i in range(len(X)):
                        # args_auxi[i]["color"] = 'C' + str((len(self.lignes_x)  + i) % 10)
                        args_auxi[i]["color"] = l_couleurs[(
                            len(self.lignes_x) + i - 1) % len(l_couleurs)]
                self.param_texts.extend(args_auxi)
        elif len(y) != len(x):
            raise (ValueError(
                "Attention : pour l'ajout d'un texte dans un Graphique, la liste des ordonnées doit être "
                "de la même taille que la liste des abscisses : x : "
                + str(len(x)) + " y : " + str(np.shape(np.array(y)))
                + str(len(np.shape(y))) + str(np.shape(np.array(y)))[1]))
        elif len(y) != len(s):
            raise (ValueError(
                "Attention : pour l'ajout d'un texte dans un Graphique, la liste des ordonnées doit être "
                "de la même taille que la liste des textes s : "
                + str(len(s)) + " y : " + str(np.shape(np.array(y)))))
        else:
            self.lignes_t_x.append(x)
            self.lignes_t_y.append(y)
            self.lignes_t_s.append(s)
            if "color" not in args:
                # args["color"] = 'C' + str(len(self.lignes_x) % 10)
                args["color"] = l_couleurs[(
                    len(self.lignes_x) - 1) % len(l_couleurs)]
            self.param_texts.append(args)

    def point(self, X: float | np.double, Y: float | np.double, marker: str = "o", **args) -> None:
        """Équivalent à plt.plot([X],[y],**args)
        Arguments :
            X,y : deux scalaire (np.double,float...)"""
        self.ligne([X], [Y], marker=marker, **args)

    def incertitudes(
            self, x, y, err_y, marker: str = "",
            échelle: str = "", **args: dict) -> None:
        """Équivalent plt.errorbar"""
        if type(err_y) is str:
            raise TypeError("Attention, une liste d'erreur est nécessaire pour l'ajout des affichage_incertitudes "
                            "dans un Graphique")
        if len(err_y) != len(x):
            raise ValueError(
                "Attention, la liste des erreur doit être de la même taille que la liste des abscisse et"
                " celle des ordonnées : x :" +
                str(len(x)) + " y : " + str(len(y)) + " err y : " + str(len(err_y)))
        if échelle == "":
            self.ligne(x, y, marker=marker, **args)
        elif échelle == "polaire":
            self.polaire(x, y, marker=marker, **args)
        elif échelle == "loglog":
            self.loglog(x, y, marker=marker, **args)
        elif échelle == "logx":
            self.logx(x, y, marker=marker, **args)
        elif échelle == "logy":
            self.logy(x, y, marker=marker, **args)
        elif échelle == "simloglog":
            self.loglog(x, y, marker=marker, **args)
        elif échelle == "simlogx":
            self.logx(x, y, marker=marker, **args)
        elif échelle == "simlogy":
            self.logy(x, y, marker=marker, **args)
        else:
            raise (ValueError("lum'échelle " + échelle + " n'est pas implémentée"))

        self.err_y[-1] = err_y
        # rappel aux fonctions dérivées d'ajout_ligne à automatiquement ajouté une liste vide dans
        # la liste des erreurs en y. On la remplace par la liste fournie en argument

    def incertitudes2(self, x, y, err_y, marker: str = "", échelle: str = "",
                      **args) -> None:
        """Équivalent à self.affichage_incertitudes, mais en remplaçant les barres d'erreur
        par un fond uniformément coloré"""

        if len(err_y) != len(x):
            raise ValueError(
                "Attention, la liste des erreur doit être de la même taille que la liste des abscisse"
                " et celle des ordonnées : x :" +
                str(len(x)) + " y : " + str(len(y)) + " err y : " + str(len(err_y)))
        if échelle == "":
            self.ligne(x, y, marker=marker, **args)
        elif échelle == "polaire":
            self.polaire(x, y, marker=marker, **args)
        elif échelle == "loglog":
            self.loglog(x, y, marker=marker, **args)
        elif échelle == "logx":
            self.logx(x, y, marker=marker, **args)
        elif échelle == "logy":
            self.logy(x, y, marker=marker, **args)
        elif échelle == "simloglog":
            self.loglog(x, y, marker=marker, **args)
        elif échelle == "simlogx":
            self.logx(x, y, marker=marker, **args)
        elif échelle == "simlogy":
            self.logy(x, y, marker=marker, **args)
        else:
            raise (ValueError("lum'échelle " + échelle + " n'est pas implémentée"))

        erry: list = list(y + err_y)
        erry2: list = list(y - err_y)
        erry2.reverse()
        erry.extend(erry2)
        x: list = list(x)
        x2: list = x.copy()
        x2.reverse()
        x.extend(x2)
        ind: np.ndarray = np.array([x, erry]).T
        self.ajout_polygone(ind, facecolor=self.param_lignes[-1]["color"])

    def polaire(self, R, Theta, marker: str = "", **args) -> None:
        """ Équivalent à plt.plot en projection polaire
Attention, ordre opposé à celui de matplotlib ; on indique le rayon puis l'angle et non l'inverse..."""
        self.ligne(R, Theta, marker, **args)
        self.config_ax(projection="polar")

    def loglog(self, x, y, marker: str = "", **args) -> None:
        """Équivalent plt.loglog"""
        self.ligne(x, y, marker, **args)
        self.config_ax(xscale="log", yscale="log")

    def simloglog(self, x, y, marker: str = "", **args) -> None:
        """Équivalent symlog en x symlog en y : affiche les valeurs positives et négatives"""
        self.ligne(x, y, marker, **args)
        self.config_ax(xscale="symlog", yscale="symlog")

    def logx(self, x, y, marker: str = "", **args) -> None:
        """Équivalent plt.semilogx"""
        self.ligne(x, y, marker, **args)
        self.config_ax(xscale="log")

    def simlogx(self, x, y, marker: str = "", **args) -> None:
        """Équivalent symlog sur l'axe x"""
        self.ligne(x, y, marker, **args)
        self.config_ax(xscale="symlog")

    def logy(self, x, y, marker: str = "", **args) -> None:
        """Équivalent `plt.semilogy`"""
        self.ligne(x, y, marker, **args)
        self.config_ax(yscale="log")

    def simlogy(self, x, y, marker: str = "", **args) -> None:
        """Équivalent symlog sur l'axe y """
        self.ligne(x, y, marker, **args)
        self.config_ax(yscale="symlog")

    def ajout_histogramme(
            self, valeurs, poids=None,
            normalisation: bool = True, statistic: str = 'sum', bins: int = 10, **args) -> None:
        if poids is None:
            poids = np.ones(len(valeurs))

        vals, bords, indices = sp.binned_statistic(
            valeurs, poids, statistic, bins, **args)

        if normalisation:
            vals /= len(valeurs)
            vals /= bords[1:] - bords[:-1]
        self.bords_histogramme.append(bords)
        self.vals_histogramme.append(vals)
        self.param_histogrammes.append(args)

    def ajout_image(self, tableau, axe_x, axe_y, **args) -> None:
        self.tableau = tableau
        self.axe_x_tableau = axe_x
        self.axe_y_tableau = axe_y
        self.param_tableau = args

    def ajout_contours(
            self, niveaux=None, contours=np.array([[]]), axe_x=None,
            axe_y=None, labels=None, labels_mask=None,**args):
        """Trace les lignes de niveaux de l'image associé à l'objet graphique
        Si niveaux est un entier n , trace les n niveaux
        Si niveaux est une liste de nombres, trace les niveaux associés à ces
        nombre
        """
        idx_niveaux: np.ndarray | None = None
        if type(niveaux) is list or type(niveaux) is np.ndarray:
            idx_niveaux = np.argsort(niveaux)
            niveaux = niveaux[idx_niveaux]

        if "colors" in args.keys() and (type(args["colors"]) is list
                                        or type(args["colors"]) is np.ndarray):
            self.couleurs_bords = args["colors"]
            del args["colors"]

        if niveaux is not None:
            args['levels'] = niveaux
            if labels is not None:
                if len(labels) != len(niveaux):
                    raise UserWarning("Graphique;ajout_niveaux : la taille des labels doit être la même que la "
                                      "taille des niveaux : niveaux",
                                      len(niveaux), "labels :", len(labels))
                self.clabels = labels[idx_niveaux]

                if labels_mask is not None:
                    if len(labels_mask) != len(niveaux):
                        raise UserWarning("Graphique;ajout_niveaux : la taille du masque des labels doit être la même "
                                          "que la taille des niveaux et des labels : niveaux/labels",
                                          len(niveaux), "labels_mask :", len(labels_mask))
                    self.clabels_mask = labels_mask[idx_niveaux]
        if len(contours) == 1:
            self.contours_est_tableau = True
            if type(args['levels']) is int:
                self.nb_contours = args['levels']
                del args['levels']
            else:
                self.nb_contours = len(args['levels'])
                self.niveaux = args['levels']
                del args['levels']
            liste_intersect: list[str] = ['alpha0', 'vmin', 'vmax', 'norm']
            if "colors" not in args:
                liste_intersect.append("cmap")
            for p in liste_intersect:
                if p in self.param_tableau:
                    self.param_contours[p] = self.param_tableau[p]
            self.param_contours.update(args)
        else:
            self.contours = contours
            self.contours_est_tableau = False
            self.axe_x_contours = axe_x
            self.axe_y_contours = axe_y
            self.param_contours = args

    def ajout_polygone(self, ind, alpha: float | np.double = 0.7, facecolor: str = 'C3', **args) -> None:
        self.indices_polygones.append(ind)
        args["alpha0"] = alpha
        args['facecolor'] = facecolor
        self.param_polygones.append(args)
        args = args.copy()
        args['color'] = facecolor
        del args['facecolor']
        if "label" in args:
            del args["label"]
        self.ligne(ind[:, 0], ind[:, 1], **args)

    def config_ax(self, **dico) -> None:
        if 'xticks' in dico:
            self.axe_x = dico['xticks']
            del dico['xticks']
        if 'yticks' in dico:
            self.axe_y = dico['yticks']
            del dico['yticks']
        if 'xticklabels' in dico:
            self.noms_axe_x = dico['xticklabels']
            del dico['xticklabels']
        if "yticklabels" in dico:
            self.noms_axe_y = dico['yticklabels']
            del dico['yticklabels']
        self.param_ax.update(dico)

    def config_légende(self, **dico) -> None:
        self.param_légende.update(dico)

    def config_bords(self, **dico) -> None:
        self.param_bords.update(dico)

    def config_fig(self, **dico) -> None:
        self.param_fig.update(dico)

    def config_enrg_fig(self, **dico) -> None:
        """Configure les paramètre de l'enregistrement de la figure"""
        self.param_enrg_fig.update(dico)

    def config_police(self, **dico) -> None:
        """Permet de gérer globalement la police
        Paramètres possibles :
            'family' : 'fantasy','monospace','sans-serif','serif','cursive'
            'styles' : 'normal', 'italic', 'oblique'
            'size' : valeur numérique
            'variants' : 'normal', 'small-caps'
            'weight' : 'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'

        """
        k: list[str] = dico.keys()
        vals: list = dico.values()
        dico: dict = {}
        for K, L in zip(k, vals):
            if "font." not in K:
                dico['font.' + K] = L
            else:
                dico[K] = L
        self.param_police.update(dico)

    def config_colorbar(self, **dico) -> None:
        """Paramètres de la colorbar"""
        self.param_colorbar.update(dico)

    def fond_noir(self) -> None:
        self.style = 'dark_background'
        for d in self.param_lignes:
            if "color" in d.keys() and d["color"] == "k":
                d["color"] = "w"
        for d in self.param_contours:
            if "color" in d.keys() and d["color"] == "k":
                d["color"] = "w"
        for d in self.param_polygones:
            if "facecolor" in d.keys() and d["facecolor"] == "k":
                d["facecolor"] = "w"

    def fond_blanc(self) -> None:
        self.style = 'default'
        for d in self.param_lignes:
            if "color" in d.keys() and d["color"] == "w":
                d["color"] = "k"
        if "colors" in self.param_contours.keys():
            for i in range(len(self.param_contours["colors"])):
                if self.param_contours["colors"] == "w":
                    self.param_contours["colors"] = "k"
        for d in self.param_polygones:
            if "facecolor" in d.keys() and d["facecolor"] == "w":
                d["facecolor"] = "k"

    def projection_lignes(self) -> None:
        # Affiche les lignes sur l'axe
        with mpl.rc_context(self.param_police):
            if self.ax is None:
                self.ax = self.fig.add_axes([0, 0, 1, 1], **self.param_ax)
            for i in range(len(self.lignes_x)):
                if len(self.param_lignes[i]) > 0:
                    if len(self.err_y[i]) > 0:
                        self.ax.errorbar(
                            self.lignes_x[i], self.lignes_y[i], self.err_y[i], **self.param_lignes[i])
                    else:
                        self.ax.plot(
                            self.lignes_x[i], self.lignes_y[i], **self.param_lignes[i])
                else:
                    self.ax.plot(self.lignes_x[i], self.lignes_y[i])
            if len(self.axe_x) == 0 or self.axe_x[0] > -np.inf:
                if len(self.axe_x) > 0:
                    self.ax.set_xlim([self.axe_x.min(), self.axe_x.max()])
                self.ax.set_xticks(self.axe_x)
            if len(self.noms_axe_x) == 0 or self.noms_axe_x[0] != "vide":
                self.ax.set_xticklabels(self.noms_axe_x)
            if len(self.axe_y) == 0 or self.axe_y[0] > -np.inf:
                if len(self.axe_y) > 0:
                    self.ax.set_ylim([self.axe_y.min(), self.axe_y.max()])
                self.ax.set_yticks(self.axe_y)
            if len(self.noms_axe_y) == 0 or self.noms_axe_y[0] != "vide":
                self.ax.set_yticklabels(self.noms_axe_y)
            if self.titre != "":
                self.ax.set_title(self.titre)

    def projection_texts(self) -> None:
        # Affiche les lignes sur l'axe
        with mpl.rc_context(self.param_police):
            if self.ax is None:
                self.ax = self.fig.add_axes([0, 0, 1, 1], **self.param_ax)
            for i in range(len(self.lignes_t_x)):
                for (X, Y, S) in zip(self.lignes_t_x[i], self.lignes_t_y[i], self.lignes_t_s[i]):
                    self.ax.text(X, Y, S, **self.param_texts[i])

    def projection_histogrammes(self) -> None:
        with mpl.rc_context(self.param_police):
            # Paramétrage de l'axe
            if self.ax is None:
                self.ax = self.fig.add_axes([0, 0, 1, 1], **self.param_ax)
            # Affichage des histogrammes
            for i in range(len(self.vals_histogramme)):
                pos = self.bords_histogramme[i][:-1]
                largeur = np.array(
                    self.bords_histogramme[i][1:]) - np.array(self.bords_histogramme[i][:-1])
                if len(self.param_histogrammes[i]) > 0:
                    self.ax.bar(x=pos, height=self.vals_histogramme[i], width=largeur, align='edge',
                                **self.param_histogrammes[i])
                else:
                    self.ax.bar(
                        x=pos, height=self.vals_histogramme[i], width=largeur, align='edge')
            # Paramétrage des coordonnées des axes
            if np.any([len(self.param_lignes[i]) > 0 for i in range(len(self.param_lignes))]):
                plt.legend(**self.param_légende)
            if len(self.axe_x) == 0 or self.axe_x[0] > -np.inf:
                if len(self.axe_x) > 0:
                    self.ax.set_xlim([self.axe_x.min(), self.axe_x.max()])
                self.ax.set_xticks(self.axe_x)
            if len(self.noms_axe_x) == 0 or self.noms_axe_x[0] != "vide":
                self.ax.set_xticklabels(self.noms_axe_x)
            if len(self.axe_y) == 0 or self.axe_y[0] > -np.inf:
                if len(self.axe_y) > 0:
                    self.ax.set_ylim([self.axe_y.min(), self.axe_y.max()])
                self.ax.set_yticks(self.axe_y)
            if len(self.noms_axe_y) == 0 or self.noms_axe_y[0] != "vide":
                self.ax.set_yticklabels(self.noms_axe_y)
            if self.titre != "":
                self.ax.set_title(self.titre)

    def projection_image(self) -> None:
        # Affiche l'image
        with mpl.rc_context(self.param_police):
            if self.ax is None:
                self.ax = self.fig.add_axes([0, 0, 1, 1], **self.param_ax)
            param_tableau: dict = self.param_tableau.copy()
            if ("scale" in self.param_colorbar.keys()
                    and self.param_colorbar["scale"] == "log"):
                if "vmin" in param_tableau.keys():
                    vmin = param_tableau["vmin"]
                    del param_tableau["vmin"]
                else:
                    vmin = self.tableau.min()
                if "vmax" in param_tableau.keys():
                    vmax = param_tableau["vmax"]
                    del param_tableau["vmax"]
                else:
                    vmax = self.tableau.max()
                if len(self.tableau.shape) == 2:
                    carte_xy = self.ax.pcolor(self.axe_x_tableau,
                                              self.axe_y_tableau, self.tableau,
                                              norm=LogNorm(vmin=vmin,
                                                           vmax=vmax),
                                              **param_tableau)
                else:
                    if "cmap" in param_tableau:
                        del param_tableau["cmap"]
                    if self.tableau.shape[2] == 3:
                        carte_xy0 = self.ax.pcolor(self.axe_x_tableau,
                                                   self.axe_y_tableau, self.tableau[:, :, 0],
                                                   norm=LogNorm(vmin=vmin,
                                                               vmax=vmax),
                                                   cmap="Reds",
                                                   **param_tableau)
                        carte_xy1 = self.ax.pcolor(self.axe_x_tableau,
                                                   self.axe_y_tableau, self.tableau[:, :, 1],
                                                   norm=LogNorm(vmin=vmin,
                                                                vmax=vmax),
                                                   cmap="Greens",
                                                   **param_tableau)
                        carte_xy2 = self.ax.pcolor(self.axe_x_tableau,
                                                   self.axe_y_tableau, self.tableau[:, :, 2],
                                                   norm=LogNorm(vmin=vmin,
                                                                vmax=vmax),
                                                   cmap="Blues",
                                                   **param_tableau)
                    elif self.tableau.shape[1] == 3:
                        carte_xy0 = self.ax.pcolor(self.axe_x_tableau,
                                                   self.axe_y_tableau, self.tableau[:, 0, :].T,
                                                   norm=LogNorm(vmin=vmin,
                                                               vmax=vmax),
                                                   cmap="Reds",
                                                   **param_tableau)
                        carte_xy1 = self.ax.pcolor(self.axe_x_tableau,
                                                   self.axe_y_tableau, self.tableau[:, 1, :].T,
                                                   norm=LogNorm(vmin=vmin,
                                                                vmax=vmax),
                                                   cmap="Greens",
                                                   **param_tableau)
                        carte_xy2 = self.ax.pcolor(self.axe_x_tableau,
                                                   self.axe_y_tableau, self.tableau[:, 2, :].T,
                                                   norm=LogNorm(vmin=vmin,
                                                                vmax=vmax),
                                                   cmap="Blues",
                                                   **param_tableau)
                    else:
                        carte_xy0 = self.ax.pcolor(self.axe_x_tableau,
                                                   self.axe_y_tableau, self.tableau[0, :, :].T,
                                                   norm=LogNorm(vmin=vmin,
                                                               vmax=vmax),
                                                   cmap="Reds",
                                                   **param_tableau)
                        carte_xy1 = self.ax.pcolor(self.axe_x_tableau,
                                                   self.axe_y_tableau, self.tableau[1, :, :].T,
                                                   norm=LogNorm(vmin=vmin,
                                                                vmax=vmax),
                                                   cmap="Greens",
                                                   **param_tableau)
                        carte_xy2 = self.ax.pcolor(self.axe_x_tableau,
                                                   self.axe_y_tableau, self.tableau[2, :, :].T,
                                                   norm=LogNorm(vmin=vmin,
                                                                vmax=vmax),
                                                   cmap="Blues",
                                                   **param_tableau)
                    # if "shading" in param_tableau:
                    #     param_tableau["interpolation"] = param_tableau["shading"]
                    #     del param_tableau["shading"]
                    #     print((np.log10(self.tableau / vmax) / np.log10(vmin / vmax)).min(),
                    #           (np.log10(self.tableau / vmax) / np.log10(vmin / vmax)).max())
                    #
                    # if "xscale" in self.param_ax and self.param_ax["xscale"] == "log":
                    #     x_int: np.ndarray = np.geomspace(np.min(self.axe_x_tableau), np.max(self.axe_x_tableau),
                    #                                      len(self.axe_x_tableau))
                    # else:
                    #     x_int: np.ndarray = np.linspace(np.min(self.axe_x_tableau), np.max(self.axe_x_tableau),
                    #                                      len(self.axe_x_tableau))
                    # if "yscale" in self.param_ax and self.param_ax["yscale"] == "log":
                    #     y_int: np.ndarray = np.geomspace(np.min(self.axe_y_tableau), np.max(self.axe_y_tableau),
                    #                                      len(self.axe_y_tableau))
                    # else:
                    #     y_int: np.ndarray = np.linspace(np.min(self.axe_y_tableau), np.max(self.axe_y_tableau),
                    #                                      len(self.axe_y_tableau))
                    # p1, p2 = np.meshgrid(self.axe_x_tableau, self.axe_y_tableau)
                    # p1, p2 = p1.flatten(), p2.flatten()
                    # points = np.array([p1, p2]).T
                    # xi_x, xi_y = np.meshgrid(x_int, y_int)
                    # tab0: np.ndarray = griddata(points, 1 - np.log10(self.tableau[:, :, 0].flatten() / vmax) / np.log10(vmin / vmax),
                    #                            xi=(xi_x, xi_y), rescale=True)
                    # tab1: np.ndarray = griddata(points, 1 - np.log10(self.tableau[:, :, 1].flatten() / vmax) / np.log10(vmin / vmax),
                    #                            xi=(xi_x, xi_y), rescale=True)
                    # tab2: np.ndarray = griddata(points, 1 - np.log10(self.tableau[:, :, 2].flatten()/ vmax) / np.log10(vmin / vmax),
                    #                            xi=(xi_x, xi_y), rescale=True)
                    # tab=np.array([tab0, tab1, tab2]).T
                    # carte_xy = self.ax.imshow(X=tab, extent=(x_int.min(), x_int.max(),
                    #                                          y_int.min(), y_int.max()), **param_tableau)
                    # self.axe_x = self.axe_x_tableau
                    # self.axe_y = self.axe_y_tableau

            else:
                if len(self.tableau.shape) == 2:
                    carte_xy = self.ax.pcolor(self.axe_x_tableau,
                                              self.axe_y_tableau, self.tableau,
                                              **param_tableau)
                else:

                    if "cmap" in param_tableau:
                        del param_tableau["cmap"]
                    carte_xy0 = self.ax.pcolor(self.axe_x_tableau,
                                               self.axe_y_tableau, self.tableau[:, :, 0],
                                               cmap="Reds",
                                               **param_tableau)
                    carte_xy1 = self.ax.pcolor(self.axe_x_tableau,
                                               self.axe_y_tableau, self.tableau[:, :, 1],
                                               cmap="Greens",
                                               **param_tableau)
                    carte_xy2 = self.ax.pcolor(self.axe_x_tableau,
                                               self.axe_y_tableau, self.tableau[:, :, 2],
                                               cmap="Blues",
                                               **param_tableau)
                    # if "shading" in param_tableau:
                    #     param_tableau["interpolation"] = param_tableau["shading"]
                    #     del param_tableau["shading"]
                    # points: np.ndarray = np.array([self.axe_x_tableau, self.axe_y_tableau]).T
                    # if "xscale" in self.param_ax and self.param_ax["xscale"] == "log":
                    #     x_int: np.ndarray = np.geomspace(np.min(self.axe_x_tableau), np.max(self.axe_x_tableau),
                    #                                      len(self.axe_x_tableau))
                    # else:
                    #     x_int: np.ndarray = np.linspace(np.min(self.axe_x_tableau), np.max(self.axe_x_tableau),
                    #                                     len(self.axe_x_tableau))
                    # if "yscale" in self.param_ax and self.param_ax["yscale"] == "log":
                    #     y_int: np.ndarray = np.geomspace(np.min(self.axe_y_tableau), np.max(self.axe_y_tableau),
                    #                                      len(self.axe_y_tableau))
                    # else:
                    #     y_int: np.ndarray = np.linspace(np.min(self.axe_y_tableau), np.max(self.axe_y_tableau),
                    #                                     len(self.axe_y_tableau))
                    # xi_x, xi_y = np.meshgrid(x_int, y_int)
                    # tab0: np.ndarray = griddata(points, self.tableau[:, :, 0],
                    #                            xi=(xi_x, xi_y), rescale=True)
                    # tab1: np.ndarray = griddata(points, self.tableau[:, :, 1],
                    #                            xi=(xi_x, xi_y), rescale=True)
                    # tab2: np.ndarray = griddata(points, self.tableau[:, :, 2],
                    #                            xi=(xi_x, xi_y), rescale=True)
                    # tab=np.array([tab0, tab1, tab2]).T
                    # carte_xy = self.ax.imshow(X=tab,
                    #                           extent=(x_int.min(), x_int.max(),
                    #                                   y_int.min(), y_int.max()),
                    #                           **param_tableau)

            params_cb: dict = self.param_colorbar.copy()
            if 'scale' in params_cb.keys():
                del params_cb['scale']
            
            if "hide" in params_cb.keys() and not params_cb["hide"]:
                del params_cb["hide"]
            if 'hide' not in params_cb.keys():
                if self.fig is None:
                    self.barre_couleurs = plt.colorbar(carte_xy,
                                                       **params_cb)
                    # self.fig.colorbar(carte_xy)
                else:
                    self.barre_couleurs = plt.colorbar(carte_xy,
                                                       **params_cb)
            if self.titre != "":
                self.ax.set_title(self.titre)

    def projection_contours(self) -> None:
        params: dict = self.param_contours.copy()
        if len(self.couleurs_bords) > 0:
            params["colors"] = self.couleurs_bords
        with mpl.rc_context(self.param_police):
            if self.ax is None:
                self.ax = self.fig.add_axes([0, 0, 1, 1], **self.param_ax)
            if len(self.niveaux) > 0:
                levels = self.niveaux  # print("levels",levels)
            else:
                levels = self.nb_contours  # print("levels",levels)
            if self.contours_est_tableau:
                if len(self.axe_x_tableau) != self.tableau.shape[1]:
                    axe_x = self.axe_x_tableau[:-1] + abs(
                        abs(self.axe_x_tableau[1:])
                        - abs(self.axe_x_tableau[:-1]))
                else:
                    axe_x = self.axe_x_tableau
                if len(self.axe_y_tableau) != self.tableau.shape[0]:
                    axe_y = self.axe_y_tableau[:-1] + abs(
                        abs(self.axe_y_tableau[1:])
                        - abs(self.axe_y_tableau[:-1]))
                else:
                    axe_y = self.axe_y_tableau

                cs = self.ax.contour(axe_x, axe_y,
                                     self.tableau, levels, **params)
            else:
                cs = self.ax.contour(self.axe_x_contours, self.axe_y_contours,
                                     self.contours, levels, **params)

            if len(self.clabels) > 0:
                dic_labels: dict = {}
                for (n, l) in zip(self.niveaux, self.clabels):
                    dic_labels[n] = l
                if len(self.clabels_mask) > 0:
                    self.ax.clabel(cs, self.niveaux[self.clabels_mask], fmt=dic_labels, **self.param_bords)
                else:
                    self.ax.clabel(cs, self.niveaux, fmt=dic_labels, **self.param_bords)
            else:
                self.ax.clabel(cs, **self.param_bords)
            if self.titre != "":
                self.ax.set_title(self.titre)

    def projection_polygones(self) -> None:
        with mpl.rc_context(self.param_police):
            for i in range(len(self.indices_polygones)):
                P = Path(self.indices_polygones[i])
                poly = PathPatch(P, **self.param_polygones[i])
                self.ax.add_patch(poly)

    def projection_figure(self) -> None:
        if self.style in plt.style.available or self.style == 'default':
            plt.style.use(self.style)
        else:
            print("Le style ", self.style, " n'est pas disponible. \n Seuls les styles :\n", plt.style.available,
                  "\n sont disponibles")
        if self.style == 'dark_background':
            self.fond_noir()
        elif self.style == 'default':
            self.fond_blanc()
        with mpl.rc_context(self.param_police):
            self.param_ax["title"] = self.titre
            if self.fig is None:
                self.fig = plt.figure()
            if len(self.param_fig) > 0:
                self.fig.set(**self.param_fig)
            if self.ax is None and "projection" in self.param_ax:
                self.ax = plt.subplot(projection=self.param_ax["projection"])
            elif self.ax is None:
                self.ax = plt.subplot()
            if len(self.param_ax) > 0:
                args = self.param_ax.copy()
                if "projection" in args:
                    del args["projection"]
                self.ax.set(**args)
            if len(self.axe_x) == 0 or self.axe_x[0] > -np.inf:
                if len(self.axe_x) > 0:
                    self.ax.set_xlim([self.axe_x.min(), self.axe_x.max()])
                self.ax.set_xticks(self.axe_x)
            if len(self.noms_axe_x) == 0 or self.noms_axe_x[0] != "vide":
                self.ax.set_xticklabels(self.noms_axe_x)
            if len(self.axe_y) == 0 or self.axe_y[0] > -np.inf:
                if len(self.axe_y) > 0:
                    self.ax.set_ylim([self.axe_y.min(), self.axe_y.max()])
                self.ax.set_yticks(self.axe_y)
            if len(self.noms_axe_y) == 0 or self.noms_axe_y[0] != "vide":
                self.ax.set_yticklabels(self.noms_axe_y)
            if self.titre != "":
                self.ax.set_title(self.titre)

            if len(self.lignes_x) > 0:
                self.projection_lignes()
            if len(self.vals_histogramme) > 0:
                self.projection_histogrammes()
            if len(self.tableau) > 1:
                self.projection_image()
            if len(self.contours) > 1 or (self.contours_est_tableau and len(self.tableau) > 1):
                self.projection_contours()
            if len(self.indices_polygones) > 0:
                self.projection_polygones()
            if len(self.lignes_t_x) > 0:
                self.projection_texts()

            if (np.any(["label" in self.param_lignes[i] for i in range(len(self.param_lignes))])
                    or np.any(["label" in self.param_histogrammes[i] for i in range(len(self.param_histogrammes))])
                    or np.any(["label" in self.param_polygones[i] for i in range(len(self.param_polygones))])):
                plt.legend(**self.param_légende)
            if self.grille:
                plt.grid()

    def enregistrement_figure(self, **args) -> None:
        args_enrg = self.param_enrg_fig.copy()
        args_enrg.update(args)
        self.projection_figure()
        plt.savefig(self.emplacement + "/" +
                    self.nom_fichier + self.ext, **args_enrg)
        self.ax = None
        plt.close()
        self.fig = None

    def affichage_figure(self) -> None:
        self.projection_figure()
        plt.show()
        self.ax = None
        self.fig = None


# ---------------------------------------------------------------------------
# --------------------------- Génération de graphs --------------------------
# ---------------------------------------------------------------------------


def ligne(x, y=None, marker: str = "", **args) -> Graphique:
    """ Équivalent plt.plot;plt.show() """
    graph: Graphique = Graphique()
    graph.ligne(x, y, marker, **args)
    graph.affichage_figure()
    return graph


def incertitudes(x, y, err_y, marker: str = "", échelle: str = "",
                 **args) -> Graphique:
    """ Équivalent plt.errorbar(x,y,err_y,marker="",**args);plt.show() """
    graph: Graphique = Graphique()
    graph.incertitudes(x, y, err_y, marker, échelle, **args)
    graph.affichage_figure()
    return graph


def incertitudes2(x, y, err_y, marker: str = "", échelle: str = "",
                  **args) -> Graphique:
    """ Équivalent affichage_incertitudes,
    mais avec un fond coloré à la place des barres d'erreur """
    graph: Graphique = Graphique()
    graph.incertitudes2(x, y, err_y, marker, échelle, **args)
    graph.affichage_figure()
    return graph


def polaire(R, Theta, **args) -> Graphique:
    """ Équivalent à plt.plot en projection polaire
    Attention, ordre opposé à celui de matplotlib :
        on indique le rayon puis l'angle et non l'inverse..."""
    graph: Graphique = Graphique()
    graph.polaire(R, Theta, **args)
    graph.affichage_figure()
    return graph


def loglog(x, y, marker: str = "", **args) -> Graphique:
    """ Équivalent plt.loglog;plt.show()
    """
    graph: Graphique = Graphique()
    graph.loglog(x, y, marker, **args)
    graph.affichage_figure()
    return graph


def simloglog(x, y, marker: str = "", **args) -> Graphique:
    """ Équivalent plt.plot;plt.show()
    avec symlog en x symlog en y : affiche les valeurs positives et négatives """
    graph: Graphique = Graphique()
    graph.simloglog(x, y, marker, **args)
    graph.affichage_figure()
    return graph


def logx(x, y, marker: str = "", **args) -> Graphique:
    """ Équivalent plt.semilogx;plt.show() """
    graph: Graphique = Graphique()
    graph.logx(x, y, marker, **args)
    graph.affichage_figure()
    return graph


def simlogx(x, y, marker: str = "", **args) -> Graphique:
    """ Équivalent plt.plot;plt.show()
    avec symlog en x : affiche les valeurs positives et négatives """
    graph: Graphique = Graphique()
    graph.simlogx(x, y, marker, **args)
    graph.affichage_figure()
    return graph


def logy(x, y, marker: str = "", **args) -> Graphique:
    """ Équivalent plt.semilogy;plt.show() """
    graph: Graphique = Graphique()
    graph.logy(x, y, marker, **args)
    graph.affichage_figure()
    return graph


def simlogy(x, y, marker: str = "", **args) -> Graphique:
    """ Équivalent `plt.plot`, `plt.show()`
    avec symlog en x : affiche les valeurs positives et négatives """
    graph: Graphique = Graphique()
    graph.simlogy(x, y, marker, **args)
    graph.affichage_figure()
    return graph


def histogramme(valeurs, poids=None, normalisation: bool = True,
                statistic: str = 'sum',
                bins: int = 10, **args) -> Graphique:
    """ Équivalent plt.hist;plt.show() mais avec des bins centrés  """
    if poids is None:
        poids = []
    graph: Graphique = Graphique()
    graph.ajout_histogramme(
        valeurs, poids, normalisation, statistic, bins, **args)
    graph.affichage_figure()
    return graph


def image(tableau, axe_x, axe_y, **args) -> Graphique:
    graph: Graphique = Graphique()
    graph.ajout_image(tableau, axe_x, axe_y, **args)
    graph.affichage_figure()
    return graph


def niveaux(x: np.ndarray | list, y: np.ndarray | list, 
            vals: np.ndarray | list, 
            npix_x: int = 400, npix_y: int = 400, logx: bool = False,
            logy: bool = False, method: str = 'cubic',
            log_vals: bool = False,
            **args):
    """Retourne une image represantant la courbe de niveau 2d associée aux 
    points définits par x,y,vals
    x: np.ndarray | list liste de taille n et dimension 1
    contenant les absiscices des points
    y: np.ndarray | list liste de taille n et dimension 1
    contenant les odronnées des points
    val: np.ndarray | list liste de taille n et dimension 1
    contenant les valeurs des points à interpoler et afficher
    npix_x: int: nombre de pixels de l'image sur l'axe x
    npix_y: int: nombre de pixels de l'image sur l'axe y
    logx: bool: indique si l'axe x est subdivisé logaritmiquement ou non
    logy: bool: indique si l'axe y est subdivisé logaritmiquement ou non
    log_val: bool: indique si les valeurs sont affichées (et extraoplées)
    sur une echelle logarithmique
    method: str: méthode d'interploation:'nearest','linear' où par défaut 
    'cubic'. Voir doc scipy.interpolate.griddata
    args: dict: dictionnaires des arguments complémentaires de images
    """
    points: np.ndarray = np.array([x, y]).T
    if logx:
        x_int: np.ndarray = np.geomspace(np.min(x), np.max(x), npix_x)
    else:        
        x_int: np.ndarray = np.linspace(np.min(x), np.max(x), npix_x)
    if logy:
        y_int: np.ndarray = np.geomspace(np.min(y), np.max(y), npix_y)
    else:        
        y_int: np.ndarray = np.linspace(np.min(y), np.max(y), npix_y)
    xi_x, xi_y = np.meshgrid(x_int, y_int)
    if log_vals:
        tab: np.ndarray = griddata(points, np.log10(vals),
                                   xi=(xi_x, xi_y), method=method, rescale=True)
        tab = 10 ** tab
        print(tab.min(), tab.max())
    else:
        tab: np.ndarray = griddata(points, vals, xi=(xi_x, xi_y), method=method)
        
    res: Graphique = image(tab, x_int, y_int, shading='nearest',
                           **args)
    res.config_colorbar(scale="log")
    return res