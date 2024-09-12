# -*- coding: utf-8 -*-

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

ii = np.iinfo(int)  # Information on numpy's integer
ii_max = ii.max  # The maximum int numerically possible

c: str = "ysqgsdkbn"  # Chain used to fill the gaps in the lists during the saving of the Graphique

# Définitions couleurs :
C1: str = '#6307ba'  # Violet / Purple
C2: str = '#16b5fa'  # Cyan
C3: str = '#2ad500'  # Vert clair / Light green
C4: str = '#145507'  # vert foncé / Dark green
C5: str = '#ff8e00'  # Orange
C6: str = '#cb0d17'  # Rouge / Red
C7: str = '#5694b2'  # Bleu pastel / Pastel blue
C8: str = '#569a57'  # Vert pastel / Pastel green
C9: str = '#b986b9'  # Lavande
C10: str = '#c6403c'  # Rouge pastel / Pastel red
C11: str = '#d39d5d'  # Beige
C12: str = '#25355d'  # Bleu / Blue
C13: str = '#fcc100'  # Jaune / Yellow
C14: str = '#7ab5fa'  # Bleu ciel / Light blue
C15: str = '#fc2700'  # Orange foncé / Dark orange
C16: str = '#0fc88f'  # Bleu-Vert / Blue-Green
C17: str = '#a8173b'  # Rouge cerise / Red
C18: str = '#1812c4'  # Bleu foncé / Dark blue
C19: str = "#000000"  # Noir / Black
C20: str = "#707070"  # Gris / Grey

l_colors: list[str] = [C1, C2, C4, C3, C5, C6, C7, C8, C9,
                       C10, C11, C12, C13, C14, C15, C16, C17, C18, C19, C20]


def linear_color_interpolation(val: np.float_ | float | list[np.float_] | list[float] | np.ndarray,
                               val_min: np.float_ = - np.inf, val_max: np.float_ = np.inf,
                               col_min: str | tuple = C1, col_max: str = C2,
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
    if isinstance(col_min, str):
        col_min: tuple = to_rgba(col_min)
    if isinstance(col_max, str):
        col_max: tuple = to_rgba(col_max)
    col_min: np.ndarray = np.array(col_min)
    col_max: np.ndarray = np.array(col_max)

    if val_min == - np.inf:
        val_min = val.min()
    if val_max == np.inf:
        val_max = val.max()

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


def lists_to_ndarray(lists_to_convert: list[list], is_scalar: bool = True) -> np.ndarray:
    """
    The objective is to convert this list of list into a numpy ndarray. To do so, all the lists of
    lists_to_merge need to be the same size. The gaps are filled with nan for numer (is_scalar==True)
    or the string c for strings
    :param lists_to_convert: the list of list
    :param is_scalar: If the content of the lists are numbers
    :return: the associated np.ndarray
    """
    if len(lists_to_convert) == 0:
        return np.array(lists_to_convert)
    lmax: int = 0
    for li in lists_to_convert:
        lmax = max(lmax, len(li))
    if lmax == 0:
        return np.array(lists_to_convert)
    res: list = []
    for i in range(len(lists_to_convert)):
        li: list = list(lists_to_convert[i])
        while len(li) < lmax:
            if is_scalar:
                li.append(np.nan)
            else:
                li.append(c)
        res.append(li)
    return np.array(res)


def ndarray_to_list(tab: np.ndarray, is_scalar: bool = True) -> list[list]:
    """
    Reconvert a list of lists from an array obtain through the 'list_to_ndarray' function
    :param tab: The array to convert
    :param is_scalar:  If the content of the lists are numbers
    :return: The list of lists
    """
    res: list = []
    for i in range(len(tab)):
        liste: list = []
        j: int = 0
        while j < len(tab[0]) and not (is_scalar and np.isnan(tab[i, j])) and tab[i, j] != c:
            liste.append(tab[i, j])
            j += 1
        res.append(np.array(liste))
    return res


# noinspection PyTypeChecker
class Graphique:
    """
The purpose of this object is to make it easier to create and manipulate graphs
It contains all the variables required to plot a graph on an axis
It can be used to display the graph, save it as a .png file and save it using a
.npy format in order to avoid having to easily regenerate the graph in a reproducible way.

#------------------------Handling procedures-------------------------------------------------------------------

To initialise a Graphique :
    - For a new Graphique: g=Graphique()
    - To open a Graphique with name n in directory f
     (n and f are two strings, by default f is the current working directory)
    e.g. f=‘../Images/Graphics’ and n=‘test_graphique’.
     (the .npz extension is not mandatory) :
        g=Graphique(n) if n is in the current directory
        g=Graphique(n,f) otherwise
To save a Graphique :
    - Assign a name to the graphic, without using an extension :
        g.filename = "new_name’
        The default name is graph_without_name. If you want to save several Graphiques in the same folder
         it is therefore important to give them a name (not automatically)
    - If necessary, assign a directory for saving:
        g.directory=‘vew_directory’ By default, the location is the current working directory.
    - To save the object :
        g.save()
    - To save the figure :
        - Assign an extension if necessary (by default the extension is svg).
         Possible extensions are those available via the matplotlib library: ‘png’, ‘pdf’, ‘svg’, etc.
            g.ext=’.new_extension'
        g.save_figure()

To show the Graphique :
    g.show()
To add a line (equivalent to plt.plot) :
    g.add_line(x, y, **args) with x and y the list(s)/ndarray to plot
     and **args are all the ather arguments of plt.plot()
    Can be repeated as many times as required
To add a histogram :
    g.add_histogram(values, weight=[], normalisation=True, statistic=‘sum’, bins=10, range=None, **args) :
    where
        - values is the array, the list of values to classify
        - weights is a list giving a possible weight for each value
        - normalisation indicates whether or not the histogram should be normalised
        - The other arguments are the same as plt.hist()
    Can be repeated as many times as necessary
To add an image :
    g.add_image(array,x_axis,y_axis,**args)
    where array represents the image to be displayed
        axis_x and axis_y give the graduations of the image axes
        **args all the other possible arguments for displaying an image
To add contours :
    g.add_contours(self, contours=np.array([[]]), x_axe=None, y_axe=None, **args) :
    - **args gives the possible arguments for plt.contours()
    - To add level lines to an image complete **args, leaving the other arguments by default
To add a polygon (coloured area delimited by a list of points) :
    g.add_polygon(ind,alpha0=0.7, facecolor=‘C3’,**args)
    with ind an array/list of dimension (n,2) where n is the number of points: ind[:,0] corresponds to the abscissas of
    the points and ind[:1] corresponds to the abscissas of the points.
    points and ind[:1] to the ordinates
    Can be repeated as many times as necessary

To go further: display several graphs in one :
    The structure does not allow such an assembly to be saved, so you need to :
    initialise a figure (plt.figure..)
    define a list of subplots (fig,axs=plt.subplot(...)
    each of these sub-figures can be associated with a graph gi :
        gi.ax=axi;gi.figure=fig
    To display the graph, call for each gi :
        gi.projection_figure() instead of g.affichage_figure()
    To display or save, call the functions plt.show() or plt.save(...)
    """

    def __init__(self, filename: str = "", directory: str = ""):
        self.filename: str = filename
        if filename == "":
            self.directory: str = "./"
            self.filename: str = "graph_without_name"
        self.ext: str = ".svg"
        self.title: str = ""
        self.style: str = 'default'  # global style:
        # styles available: plt.style.available : 'default' 'Solarize_Light2'
        # '_classic_test_patch' '_mpl-gallery' '_mpl-gallery-nogrid'
        # 'bmh' 'classic' 'dark_background' 'fast' 'fivethirtyeight' 'ggplot'
        # 'grayscale' 'seaborn' 'seaborn-bright' 'seaborn-colorblind' 'seaborn-dark' 'seaborn-dark-palette'
        # 'seaborn-darkgrid' 'seaborn-deep' 'seaborn-muted' 'seaborn-notebook' 'seaborn-paper' 'seaborn-pastel'
        # 'seaborn-poster' 'seaborn-talk' 'seaborn-ticks' 'seaborn-white' 'seaborn-whitegrid' 'tableau-colorblind10'
        self.param_font: dict = {
            "font.size": 13}  # Contains additional parameters for global font management
        self.ax: axes = None  # Contains additional parameters for  ax ex : xlabel='x',xscale='log'...
        self.param_ax: dict = dict()
        self.colorbar: list[Colorbar] = None
        self.param_colorbar: list[dict] = [dict()]
        self.ticks_colorbar: list[np.ndarray | list] = [[]]
        # Contains additional parameters for the colorbars ex : label="legende"...
        # The first one is automatically associated with the image
        self.custum_colorbar_colors: list[list] = None
        # To bild custums discrete colorbars defind by the colors in thoses list (one list for one colorbar)
        self.custum_colorbar_values: list[list] = None
        # The values associated with the colors of self.custum_colorbar_colors

        self.fig: Figure = None
        self.param_fig: dict = dict()  # Contains additional parameters for the Figure ex : facecolor="w"...
        self.param_enrg_fig: dict = dict(bbox_inches="tight")
        # Contains additional parameters to save the Figure ex : dpi=300...
        
        # liste of position of x-axis ticks
        self.x_axe: np.ndarray[float] = np.array([-np.inf])
        # List of x-ticks labels if not "empty"
        self.labels_x_ticks: np.ndarray[str] = np.array(["empty"])

        # liste of position of y-axis ticks
        self.y_axe: np.ndarray[float] = np.array([-np.inf])
        # List of y-ticks labels if not "empty"
        self.labels_y_ticks: np.ndarray[str] = np.array(["empty"])
        self.lines_x: list[
            list[float | np.float_]] = []  # list containing lists of x coordinates to be displayed via plt.plot
        self.lines_y: list[
            list[float | np.float_]] = []  # list containing lists of y coordinates to be displayed via plt.plot
        self.lines_t_x: list[
            list[float | np.float_]] = []  # list containing lists of x coordinates to be displayed via plt.text
        self.lines_t_y: list[
            list[float | np.float_]] = []  # list containing lists of y coordinates to be displayed via plt.text
        self.lines_t_s: list[
            list[float | np.float_]] = []  # list containing lists of string to be displayed via plt.text
        self.err_y: list[list[
            float | np.float_]] = []  # list containing lists of errors associated with y coordinates

        # Additianal parameter's dictionarys
        self.param_lines: list = []  # for lines
        self.param_texts: list = []  # for texts

        # histogrammes
        self.bords_histogramme: list = []  # List of lists of coordinates of the borders of histograms' bars
        # liste contenant les valeurs des hauteurs de barres des histogrammes
        self.vals_histogramme: list = []  # List of lists of the values of each histograms' bars
        self.param_histogrammes: list = []  # Additional parameters dictionary for plt.bar

        self.param_legende: dict = dict()  # parameters of plt.legende

        # Table to plot via plt.pcolor, the limites are x_axe/y
        self.array_image: np.ndarray = np.array([[]])
        self.x_axe_image: np.ndarray = np.array([])  # list of x-coordinate for the image
        self.y_axe_image: np.ndarray = np.array([])  # list of y-coordinate for the image

        # Parameters of pcolor colorbar,legend...
        self.param_image: dict = dict()  # for pcolor
        self.array_contours: np.ndarray = np.array([[]])
        # table to plot via contour, the limites are x_axe/y switch of half a bin

        self.clabels: np.ndarray = np.array([])  # Label to plot for each contours levels
        self.clabels_mask: np.ndarray = np.array([])  # Mask or index list of labels to plot for contours
        self.param_contours: dict = dict()  # parameter of contours : alpha0,label...
        self.param_labels_contours: dict = dict(fontsize=15, inline=True)
        # Contains additional parameters for the labels of the contours
        self.color_label_contours: list[str] = []
        self.x_axe_contours: np.ndarray = np.array([])  # liste of x-coordinate for contours
        self.y_axe_contours: np.ndarray = np.array([])   # liste of y-coordinate for contours
        # lists of levels for contours to plot
        self.levels: np.ndarray = np.array([])
        # number of levels for contours to plot, ignore if levels != np.array([])
        self.nb_contours: int = 10
        self.tab_contours_is_image: bool = False  # If True, the table used for contour is self.image

        self.index_polygons: list = []  # Index of polygone polygons to plot
        self.param_polygons: list = []  # Parameters for the polygons plotting
        self.grid: bool = False  # If True add a background grid via plt.grid()
        if filename != "":
            values_to_load: np.lib.npyio.NpzFile = np.load(filename + ".npz")
            self.filename = filename
            if directory != "":
                self.directory = directory
            elif "/" in directory:
                i: int = directory.find("/")
                while directory.find("/") > 0:
                    # Cherche la dernière occurrence de "/"
                    i = directory.find("/")
                self.directory = filename[:i]
                self.filename = filename[i:]
            else:
                self.directory = "./"
            if "ext" in values_to_load:
                self.ext = str(values_to_load["ext"])
            self.ax = None
            self.param_ax = dict()
            if 'style' in values_to_load:
                self.style = str(values_to_load["style"])
            if "param_colorbar" in values_to_load:
                param_colorbar = ndarray_to_list(
                    values_to_load["param_colorbar"], is_scalar=False)
                names_param_colorbar = ndarray_to_list(
                    values_to_load["name_param_colorbar"], is_scalar=False)
                for i in range(len(param_colorbar)):
                    self.param_colorbar.append(dict())
                    for j in range(len(param_colorbar[i])):
                        self.param_colorbar[i][names_param_colorbar[i][j]] = param_colorbar[i][j]
                        try:
                            self.param_colorbar[i][names_param_colorbar[i][j]] = float(
                                param_colorbar[i][j])
                        except ValueError:
                            try:
                                self.param_colorbar[i][names_param_colorbar[i][j]] = str(
                                    param_colorbar[i][j])
                            except ValueError:
                                self.param_colorbar[i][names_param_colorbar[i][j]] = param_colorbar[i][j]

            if "ticks_colorbar" in values_to_load:
                self.ticks_colorbar = ndarray_to_list(values_to_load["ticks_colorbar"], is_scalar=True)
            if "custum_colorbar_colors" in values_to_load:
                self.custum_colorbar_colors = ndarray_to_list(values_to_load["custum_colorbar_colors"], is_scalar=False)
            if "custum_colorbar_values" in values_to_load:
                self.custum_colorbar_values = ndarray_to_list(values_to_load["custum_colorbar_values"], is_scalar=True)

            if "param_labels_contours" in values_to_load:
                if len(values_to_load["name_param_labels_contours"].shape) != 1:
                    print("error during the reading of labels_contours parameters")
                for i in range(len(values_to_load["name_param_labels_contours"])):
                    self.param_labels_contours[values_to_load["name_param_labels_contours"][i]] = (
                        values_to_load)["param_labels_contours"][i]
                    try:
                        self.param_labels_contours[values_to_load["name_param_labels_contours"][i]] = float(
                            values_to_load["param_labels_contours"][i])
                    except ValueError:
                        try:
                            self.param_labels_contours[values_to_load["name_param_labels_contours"][i]] = str(
                                values_to_load["param_labels_contours"][i])
                        except ValueError:
                            self.param_labels_contours[values_to_load["name_param_labels_contours"][i]] =\
                                values_to_load["param_labels_contours"][i]
            if "color_label_contours" in values_to_load:
                self.color_label_contours = values_to_load["color_label_contours"]
            if "param_font" in values_to_load:
                if len(values_to_load["name_param_font"].shape) != 1:
                    print("error during the reading of font parameters")
                for i in range(len(values_to_load["name_param_font"])):
                    self.param_font[values_to_load["name_param_font"][i]] = values_to_load["param_font"][i]
                    try:
                        self.param_font[values_to_load["name_param_font"][i]] = float(values_to_load["param_font"][i])
                    except ValueError:
                        try:
                            self.param_font[values_to_load["name_param_font"][i]] = str(values_to_load["param_font"][i])
                        except ValueError:
                            self.param_font[values_to_load["name_param_font"][i]] = (
                                values_to_load)["param_font"][i]
            if "param_ax" in values_to_load:
                if len(values_to_load["name_param_ax"].shape) != 1:
                    print("error during the reading of ax parameters")
                for i in range(len(values_to_load["name_param_ax"])):
                    self.param_ax[values_to_load["name_param_ax"][i]] = values_to_load["param_ax"][i]
                    try:
                        self.param_ax[values_to_load["name_param_ax"][i]] = float(values_to_load["param_ax"][i])
                    except ValueError:
                        try:
                            self.param_ax[values_to_load["name_param_ax"][i]] = str(values_to_load["param_ax"][i])
                        except ValueError:
                            self.param_ax[values_to_load["name_param_ax"][i]] = values_to_load["param_ax"][i]
            self.colorbar = None
            if "param_fig" in values_to_load:
                if len(values_to_load["name_param_fig"].shape) != 1:
                    print("error during the reading of Figure parameters")
                for i in range(len(values_to_load["name_param_fig"])):
                    self.param_fig[values_to_load["name_param_fig"][i]] = values_to_load["param_fig"][i]
                    try:
                        self.param_fig[values_to_load["name_param_fig"][i]] = float(values_to_load["param_fig"][i])
                    except ValueError:
                        try:
                            self.param_fig[values_to_load["name_param_fig"][i]] = str(
                                values_to_load["param_fig"][i])
                        except ValueError:
                            self.param_fig[values_to_load["name_param_fig"][i]] = values_to_load["param_fig"][i]
            if "param_enrg_fig" in values_to_load:
                if len(values_to_load["name_param_enrg_fig"].shape) != 1:
                    print("error during the reading of Figure saving parameters")
                for i in range(len(values_to_load["name_param_enrg_fig"])):
                    self.param_enrg_fig[values_to_load["name_param_enrg_fig"][i]] = values_to_load["param_enrg_fig"][i]
                    try:
                        self.param_enrg_fig[values_to_load["name_param_enrg_fig"][i]] = float(
                            values_to_load["param_enrg_fig"][i])
                    except ValueError:
                        try:
                            self.param_enrg_fig[values_to_load["name_param_enrg_fig"][i]] = str(
                                values_to_load["param_enrg_fig"][i])
                        except ValueError:
                            self.param_enrg_fig[values_to_load["name_param_enrg_fig"][i]] =\
                                values_to_load["param_enrg_fig"][i]
            if "x_axe" in values_to_load:
                self.x_axe = values_to_load["x_axe"]
            if "labels_x_ticks" in values_to_load:
                self.labels_x_ticks = values_to_load["labels_x_ticks"]
            if "y_axe" in values_to_load:
                self.y_axe = values_to_load["y_axe"]
            if "labels_y_ticks" in values_to_load:
                self.labels_y_ticks = values_to_load["labels_y_ticks"]
            if "lines_x" in values_to_load:
                self.lines_x = ndarray_to_list(values_to_load["lines_x"], is_scalar=True)
            if "lines_y" in values_to_load:
                self.lines_y = ndarray_to_list(values_to_load["lines_y"], is_scalar=True)
            if "lines_t_x" in values_to_load:
                self.lines_t_x = ndarray_to_list(
                    values_to_load["lines_t_x"], is_scalar=True)
            if "lines_t_y" in values_to_load:
                self.lines_t_y = ndarray_to_list(
                    values_to_load["lines_t_y"], is_scalar=True)
            if "lines_t_s" in values_to_load:
                self.lines_t_s = ndarray_to_list(
                    values_to_load["lines_t_s"], is_scalar=False)
            if "err_y" in values_to_load:
                self.err_y = ndarray_to_list(values_to_load["err_y"], is_scalar=True)
                
            self.param_lines = []
            if "param_lines" in values_to_load:
                param_lines = ndarray_to_list(
                    values_to_load["param_lines"], is_scalar=False)
                names_param_lines = ndarray_to_list(
                    values_to_load["name_param_lines"], is_scalar=False)
                for i in range(len(param_lines)):
                    self.param_lines.append(dict())
                    for j in range(len(param_lines[i])):
                        self.param_lines[i][names_param_lines[i][j]] = param_lines[i][j]
                        try:
                            self.param_lines[i][names_param_lines[i][j]] = float(
                                param_lines[i][j])
                        except ValueError:
                            try:
                                self.param_lines[i][names_param_lines[i][j]] = str(
                                    param_lines[i][j])
                            except ValueError:
                                self.param_lines[i][names_param_lines[i][j]] = param_lines[i][j]
                                
            self.param_texts = []
            if "param_texts" in values_to_load:
                param_texts = ndarray_to_list(
                    values_to_load["param_texts"], is_scalar=False)
                names_param_texts = ndarray_to_list(
                    values_to_load["name_param_texts"], is_scalar=False)
                for i in range(len(param_texts)):
                    self.param_texts.append(dict())
                    for j in range(len(param_texts[i])):
                        self.param_texts[i][names_param_texts[i][j]] = param_texts[i][j]
                        try:
                            self.param_texts[i][names_param_texts[i][j]] = float(
                                param_texts[i][j])
                        except ValueError:
                            try:
                                self.param_texts[i][names_param_texts[i][j]] = str(
                                    param_texts[i][j])
                            except ValueError:
                                self.param_texts[i][names_param_texts[i][j]] = param_texts[i][j]

            if "bords_histogramme" in values_to_load:
                self.bords_histogramme = ndarray_to_list(
                    values_to_load["bords_histogramme"], is_scalar=True)
            if "vals_histogramme" in values_to_load:
                self.vals_histogramme = ndarray_to_list(values_to_load["vals_histogramme"],
                                                        is_scalar=True)
            self.param_histogrammes = []
            if "param_histogrammes" in values_to_load:
                param_histogrammes: list[np.ndarray] = ndarray_to_list(
                    values_to_load["param_histogrammes"], is_scalar=False)
                names_param_histogrammes: list[np.ndarray[str]] =\
                    ndarray_to_list(values_to_load["name_param_histogrammes"], is_scalar=False)
                for i in range(len(param_histogrammes)):
                    self.param_histogrammes.append(dict())
                    for j in range(len(param_histogrammes[i])):
                        self.param_histogrammes[i][names_param_histogrammes[i][j]] = param_histogrammes[i][j]
                        try:
                            self.param_histogrammes[i][names_param_histogrammes[i][j]] = float(
                                param_histogrammes[i][j])
                        except ValueError:
                            try:
                                self.param_histogrammes[i][names_param_histogrammes[i][j]] = str(
                                    param_histogrammes[i][j])
                            except ValueError:
                                self.param_histogrammes[i][names_param_histogrammes[i][j]] = param_histogrammes[i][j]

            if "param_legende" in values_to_load:
                for i in range(len(values_to_load["name_param_legende"])):
                    self.param_legende[values_to_load["name_param_legende"][i]] = values_to_load["param_legende"][i]
                    try:
                        self.param_legende[values_to_load["name_param_legende"][i]] = float(
                            values_to_load["param_legende"][i])
                    except ValueError:
                        try:
                            self.param_legende[values_to_load["name_param_legende"][i]] = str(
                                values_to_load["param_legende"][i])
                        except ValueError:
                            self.param_legende[values_to_load["name_param_legende"][i]] =\
                                values_to_load["param_legende"][i]

            if "image" in values_to_load:                
                self.array_image = values_to_load["image"]
                self.param_image = dict()
                if len(values_to_load["param_image"].shape) != 1:
                    print("error during the reading of image parameters")
                for i in range(len(values_to_load["param_image"])):
                    self.param_image[values_to_load["name_param_image"][i]] = values_to_load["param_image"][i]
                    try:
                        self.param_image[values_to_load["name_param_image"][i]] = float(
                            values_to_load["param_image"][i])
                    except ValueError:
                        try:
                            self.param_image[values_to_load["name_param_image"][i]] = str(
                                values_to_load["param_image"][i])
                        except ValueError:
                            self.param_image[values_to_load["name_param_image"][i]] = values_to_load["param_image"][i]
                self.x_axe_image = values_to_load["x_axe_image"]
                # liste des coordonnées y de contours
                self.y_axe_image = values_to_load["y_axe_image"]
            if "contours" in values_to_load:
                self.array_contours = values_to_load["contours"] 
                self.x_axe_contours = values_to_load["x_axe_contours"]
                self.y_axe_contours = values_to_load["y_axe_contours"]
            if "param_contours" in values_to_load:
                self.param_contours = dict()
                if len(values_to_load["param_contours"].shape) != 1:
                    print("error during the reading of contours parameters")
                for i in range(len(values_to_load["param_contours"])):
                    self.param_contours[values_to_load["name_param_contours"][i]] = values_to_load["param_contours"][i]
                    try:
                        self.param_contours[values_to_load["name_param_contours"][i]] = float(
                            values_to_load["param_contours"][i])
                    except ValueError:
                        try:
                            self.param_contours[values_to_load["name_param_contours"][i]] = str(
                                values_to_load["param_contours"][i])
                        except ValueError:
                            self.param_contours[values_to_load["name_param_contours"][i]] = (
                                values_to_load)["param_contours"][i]

            if "levels" in values_to_load:
                self.levels = values_to_load["levels"]
                if "clabels" in values_to_load:
                    self.clabels = values_to_load["clabels"]
                if "clabels_mask" in values_to_load:
                    self.clabels_mask = values_to_load["clabels_mask"]
            if "parameters" in values_to_load:
                self.title = values_to_load["parameters"][0]
                self.nb_contours = int(values_to_load["parameters"][1])
                self.tab_contours_is_image = bool(int(values_to_load["parameters"][2]))
                self.grid = bool(int(values_to_load["parameters"][3]))
            if "index_polygons" in values_to_load:
                ind1 = ndarray_to_list(
                    values_to_load["index_polygons_1"], is_scalar=True)
                ind2 = ndarray_to_list(
                    values_to_load["index_polygons_2"], is_scalar=True)
                for i in range(len(ind1)):
                    self.index_polygons.append(
                        np.array([ind1[i], ind2[i]]).T)
                param_polygons = ndarray_to_list(
                    values_to_load["param_polygons"], is_scalar=False)
                name_param_polygons = ndarray_to_list(
                    values_to_load["name_param_polygons"], is_scalar=False)
                for i in range(len(param_polygons)):
                    self.param_polygons.append(dict())
                    for j in range(len(param_polygons[i])):
                        self.param_polygons[i][name_param_polygons[i][j]] = param_polygons[i][j]
                        try:
                            self.param_polygons[i][name_param_polygons[i][j]] = float(
                                param_polygons[i][j])
                        except ValueError:
                            try:
                                self.param_polygons[i][name_param_polygons[i][j]] = str(
                                    param_polygons[i][j])
                            except ValueError:
                                self.param_polygons[i][name_param_polygons[i][j]] = param_polygons[i][j]
            values_to_load.close()

    def save(self, filename: str = "graph_without_name", directory: str = None) -> None:
        """
        Save the Graphique in self.directory (default the current working directory) in npz compress
        format.
        :param filename: The name of the .npz file (default: "graph_without_name")
        :param directory: Graphique's directory (default self.directory (default : the curent working directory))
        :return: None
        """
        if filename != "graph_without_name":
            if ".npz" in filename:
                self.filename = filename[:-4]
            else:
                self.filename = filename
        if directory is not None:
            self.directory = directory
        enrg: dict = dict()  # Dictionary containing all the necessary information :
        # Used like :  np.savez_compressed(name_fichier,**enrg)
        
        enrg["ext"] = self.ext
        enrg["style"] = self.style
        if len(self.param_colorbar) > 0:
            param_colorbar: list = []
            name_param_colorbar: list[list[str]] = []
            for i in range(len(self.param_colorbar)):
                param_colorbar.append([]), name_param_colorbar.append([])
                for key in self.param_colorbar[i].keys():
                    param_colorbar[i].append(self.param_colorbar[i][key])
                    name_param_colorbar[i].append(key)
            enrg["param_colorbar"] = lists_to_ndarray(param_colorbar, is_scalar=False)
            enrg["name_param_colorbar"] = lists_to_ndarray(
                name_param_colorbar, is_scalar=False)
        if len(self.ticks_colorbar) > 0:
            enrg["ticks_colorbar"] = lists_to_ndarray(self.ticks_colorbar, is_scalar=True)

        if self.custum_colorbar_colors is not None:
            enrg["custum_colorbar_colors"] = lists_to_ndarray(self.custum_colorbar_colors,
                                                              is_scalar=False)
        if self.custum_colorbar_values is not None:
            enrg["custum_colorbar_values"] = lists_to_ndarray(self.custum_colorbar_values, is_scalar=True)

        if len(self.param_labels_contours) > 0:
            param_labels_contours: list = []
            name_param_labels_contours: list[str] = []
            for key in self.param_labels_contours.keys():
                param_labels_contours.append(self.param_labels_contours[key])
                name_param_labels_contours.append(key)
            enrg["param_labels_contours"] = np.array(param_labels_contours)
            enrg["name_param_labels_contours"] = np.array(name_param_labels_contours)
        if len(self.param_font) > 0:
            param_font: list = []
            name_param_font: list[str] = []
            for key in self.param_font.keys():
                param_font.append(self.param_font[key])
                name_param_font.append(key)
            enrg["param_font"] = np.array(param_font)
            enrg["name_param_font"] = np.array(name_param_font)
        if len(self.param_ax) > 0:
            param_ax: list = []
            name_param_ax: list[str] = []
            for key in self.param_ax.keys():
                param_ax.append(self.param_ax[key])
                name_param_ax.append(key)
            enrg["param_ax"] = np.array(param_ax)
            enrg["name_param_ax"] = np.array(name_param_ax)
        if len(self.param_fig) > 0:
            param_fig: list = []
            name_param_fig: list[str] = []
            for key in self.param_fig.keys():
                param_fig.append(self.param_fig[key])
                name_param_fig.append(key)
            enrg["param_fig"] = np.array(param_fig)
            enrg["name_param_fig"] = np.array(name_param_fig)
        if len(self.param_enrg_fig) > 0:
            param_enrg_fig: list = []
            name_param_enrg_fig: list[str] = []
            for key in self.param_enrg_fig.keys():
                param_enrg_fig.append(self.param_enrg_fig[key])
                name_param_enrg_fig.append(key)
            enrg["param_enrg_fig"] = np.array(param_enrg_fig)
            enrg["name_param_enrg_fig"] = np.array(name_param_enrg_fig)
        if len(self.x_axe) == 0 or self.x_axe[0] > -np.inf:
            enrg["x_axe"] = self.x_axe
        if len(self.labels_x_ticks) == 0 or self.labels_x_ticks[0] != "empty":
            enrg["labels_x_ticks"] = self.labels_x_ticks
        if len(self.y_axe) == 0 or self.y_axe[0] > -np.inf:
            enrg["y_axe"] = self.y_axe
        if len(self.labels_y_ticks) == 0 or self.labels_y_ticks[0] != "empty":
            enrg["labels_y_ticks"] = self.labels_y_ticks
        if len(self.lines_x) > 0:
            enrg["lines_x"] = lists_to_ndarray(self.lines_x, is_scalar=True)
        if len(self.lines_y) > 0:
            enrg["lines_y"] = lists_to_ndarray(self.lines_y, is_scalar=True)
        if len(self.lines_t_x) > 0:
            enrg["lines_t_x"] = lists_to_ndarray(self.lines_t_x, is_scalar=True)
        if len(self.lines_t_y) > 0:
            enrg["lines_t_y"] = lists_to_ndarray(self.lines_t_y, is_scalar=True)
        if len(self.lines_t_s) > 0:
            enrg["lines_t_s"] = lists_to_ndarray(self.lines_t_s)
        if len(self.err_y) > 0:
            enrg["err_y"] = lists_to_ndarray(self.err_y, is_scalar=True)
        if len(self.param_lines) > 0:
            param_lines: list = []
            name_param_lines: list[list[str]] = []
            for i in range(len(self.param_lines)):
                param_lines.append([]), name_param_lines.append([])
                for key in self.param_lines[i].keys():
                    param_lines[i].append(self.param_lines[i][key])
                    name_param_lines[i].append(key)
            enrg["param_lines"] = lists_to_ndarray(param_lines, is_scalar=False)
            enrg["name_param_lines"] = lists_to_ndarray(
                name_param_lines, is_scalar=False)
        if len(self.param_texts) > 0:
            param_texts: list = []
            name_param_texts: list[list[str]] = []
            for i in range(len(self.param_texts)):
                param_texts.append([]), name_param_texts.append([])
                for key in self.param_texts[i].keys():
                    param_texts[i].append(self.param_texts[i][key])
                    name_param_texts[i].append(key)
            enrg["param_texts"] = lists_to_ndarray(param_texts, is_scalar=False)
            enrg["name_param_texts"] = lists_to_ndarray(
                name_param_texts, is_scalar=False)
        if len(self.bords_histogramme) > 0:
            enrg["bords_histogramme"] = lists_to_ndarray(
                self.bords_histogramme, is_scalar=True)
        if len(self.vals_histogramme) > 0:
            enrg["vals_histogramme"] = lists_to_ndarray(
                self.vals_histogramme, is_scalar=True)
        if len(self.param_histogrammes) > 0:
            param_histogrammes: list = []
            name_param_histogrammes: list[list[str]] = []
            for i in range(len(self.param_histogrammes)):
                param_histogrammes.append([])
                name_param_histogrammes.append([])
                for key in self.param_histogrammes[i].keys():
                    param_histogrammes[i].append(
                        self.param_histogrammes[i][key])
                    name_param_histogrammes[i].append(key)
            enrg["param_histogrammes"] = lists_to_ndarray(
                param_histogrammes, is_scalar=False)
            enrg["name_param_histogrammes"] = lists_to_ndarray(
                name_param_histogrammes, is_scalar=False)
        if len(self.param_legende) > 0:
            param_legende: list = []
            name_param_legende: list[str] = []
            for key in self.param_legende.keys():
                param_legende.append(self.param_legende[key])
                name_param_legende.append(key)
            enrg["param_legende"] = np.array(param_legende)
            enrg["name_param_legende"] = np.array(name_param_legende)
        if len(self.array_image) > 1:
            enrg["tableau"] = self.array_image
            enrg["x_axe_tableau"] = self.x_axe_image
            enrg["y_axe_tableau"] = self.y_axe_image
            param_tableau: list = []
            name_param_tableau: list[str] = []
            for key in self.param_image.keys():
                param_tableau.append(self.param_image[key])
                name_param_tableau.append(key)
            print(param_tableau, name_param_tableau)
            enrg["param_tableau"] = np.array(param_tableau)
            enrg["name_param_tableau"] = np.array(name_param_tableau)
        if len(self.array_contours) > 1:
            enrg["contours"] = self.array_contours
            enrg["x_axe_contours"] = self.x_axe_contours
            enrg["y_axe_contours"] = self.y_axe_contours
        if len(self.color_label_contours) > 0:
            enrg["color_label_contours"] = self.color_label_contours
        if len(self.param_contours) > 0:
            param_contours: list = []
            name_param_contours: list[str] = []
            for key in self.param_contours.keys():
                param_contours.append(self.param_contours[key])
                name_param_contours.append(key)
            enrg["param_contours"] = np.array(param_contours)
            enrg["name_param_contours"] = np.array(name_param_contours)
        if len(self.levels) > 0:
            enrg["levels"] = self.levels
            if len(self.clabels) > 0:
                enrg["clabels"] = self.clabels
            if len(self.clabels_mask) > 0:
                enrg["clabels_mask"] = self.clabels_mask
        param = [self.title, str(self.nb_contours), str(
            int(self.tab_contours_is_image)), str(int(self.grid))]
        enrg["parameters"] = param
        if len(self.index_polygons) > 0:
            ind1: list[float | np.double] = []
            ind2: list[float | np.double] = []
            for liste in self.index_polygons:
                ind1.append(liste.T[0])
            for liste in self.index_polygons:
                ind2.append(liste.T[1])
            enrg["index_polygons_1"] = lists_to_ndarray(ind1, is_scalar=True)
            enrg["index_polygons_2"] = lists_to_ndarray(ind2, is_scalar=True)
            param_polygons: list = []
            name_param_polygons: list[list[str]] = []
            for i in range(len(self.param_polygons)):
                param_polygons.append([])
                name_param_polygons.append([])
                for key in self.param_polygons[i].keys():
                    param_polygons[i].append(self.param_polygons[i][key])
                    name_param_polygons[i].append(key)
            enrg["param_polygons"] = lists_to_ndarray(
                param_polygons, is_scalar=False)
            enrg["name_param_polygons"] = lists_to_ndarray(
                name_param_polygons, is_scalar=False)
        if ".npz" not in self.filename:
            if self.directory[-1] == "/":
                np.savez_compressed(self.directory +
                                    self.filename + ".npz", **enrg)
            else:
                np.savez_compressed(self.directory + "/" +
                                    self.filename + ".npz", **enrg)
        else:
            if self.directory[-1] == "/":
                np.savez_compressed(self.directory +
                                    self.filename, **enrg)
            else:
                np.savez_compressed(self.directory + "/" +
                                    self.filename, **enrg)

    def customized_cmap(self, values: list[np.float_] | np.ndarray | tuple,
                        colors: list | np.ndarray | tuple | None = None,
                        ticks: list | np.ndarray[np.float_] | None= None,
                        **kwargs) -> None:
        """
        Build a customized discrete colorbar
        :param values: The values of the colormap's color intervals if len(values)==2, the interval is automatically
            defined as a linear subdivision of the interval between values[0] and values[1] of size 255
        :param colors: The associated colors, if None, a linear variation beteween C1 and C2 is bild
        :param ticks: Array of ticks for the colorbar If None, ticks are determined automatically from the input.
        :param kwargs: Additionals arguments for the colorbar
        :return: None
        """
        if len(values) < 2:
            raise UserWarning("Graphique.custumized_cmap : the len of values need to be higer than 2")
        if isinstance(values, tuple) or len(values) == 2:
            values = np.linspace(values[0], values[1], 255)
            if colors is not None and (isinstance(colors, tuple) or len(colors) == 2):
                colors = linear_color_interpolation(values, col_min=colors[0], col_max=colors[1])
        elif colors is not None and (isinstance(colors, tuple) or len(colors) == 2):
            colors = linear_color_interpolation(values, col_min=colors[0], col_max=colors[1])
        if colors is not None and len(colors) != len(values):
            raise UserWarning("Graphique.custumized_cmap : values and colors need to have the same size :"
                              "len(values)=", len(values), " len(colors)=", len(colors))
        if colors is None:
            colors = linear_color_interpolation(values, col_min=C1, col_max=C2)
        if self.custum_colorbar_values is not None:
            self.custum_colorbar_values.append(values)
        else:
            self.custum_colorbar_values = [values]
        if self.custum_colorbar_colors is not None:
            self.custum_colorbar_colors.append(colors)
        else:
            self.custum_colorbar_colors = [colors]
        if ticks is None:
            self.ticks_colorbar.append([])
        else:
            self.ticks_colorbar.append(ticks)
        self.param_colorbar.append(kwargs)

    def line(self, x: np.ndarray | list, y: np.ndarray | list = None,
             marker: str | list = "", **args) -> None:
        """
        Equivalent to plt.plot with a small improvement:
        if y is in two dimensions, the second dimension is plotted :
            - `self.line(x,[y1,y2], *args)` is equivalent to
                `'`plt.plot(x, y1, *args)
                plt.plot(x, y2, *args)`
            - if y1 and y2 have not the same size:
                `self.line([x1,x2],[y1, y2], *args)`
            - If others arguments are list of the same size of x and y, they are also split :
                `self.line((x1, x2], [y1, y2], marker=".", label=["Curve1", "Curve2"]`
             is equivalent to
                `plt.plot(x, y1, marker=".", label="Curve1")
                plt.plot(x, y2, marker=".", label="Curve2")`
        :param x: Abscissa(s)
        :param y: Ordinate(s)
        :param marker: The marker (default="" (no marker), ex ".", ",", "o", "v"...
         see matplotlib documentation for all the possibility
        :param args: Additional argument to plot() function like linestyle, color....
        :return: None
        """
        
        if type(y) is str:
            marker = y
            y = None
        if y is None:
            y = np.copy(x)
            x = np.arange(0, len(y))
        if isinstance(x[0], list) | isinstance(x[0], np.ndarray):
            if isinstance(x[0][0], list) | isinstance(x[0][0], np.ndarray):
                raise UserWarning("Graphique.line the x-axis dimension cannot be superior than 2")
        if isinstance(x[0], list) | isinstance(x[0], np.ndarray):
            if isinstance(x[0][0], list) | isinstance(x[0][0], np.ndarray):
                raise UserWarning("Graphique.line the x-axis dimension cannot be superior than 2")
            else:
                dim_x: int = 2
        else:
            dim_x: int = 1
        if isinstance(y[0], list) | isinstance(y[0], np.ndarray):
            if isinstance(y[0][0], list) | isinstance(y[0][0], np.ndarray):
                raise  UserWarning("Graphique.line the y-axis dimension cannot be superior than 2")

            else:
                dim_y: int = 2
        else:
            dim_y: int = 1
        if (dim_x == 2 and dim_y == 2 and
                np.any(np.array([len(X) != len(Y) for (X, Y) in zip(x, y)]))):
            raise UserWarning("Graphique.line : the sizes of arrays of the abscissa "
                              "doesn't mach with the sizes of the array of ordinates : ",
                              [(len(X), len(Y)) for (X, Y) in zip(x, y)])
        elif (dim_y == 2 and dim_x == 1 and
              np.any(np.array([len(x) != len(Y) for Y in y]))):
            raise UserWarning("Graphique.line : the sizes of arrays of the abscissa "
                              "doesn't mach with the sizes of the array of ordinates : ",
                              [(len(x), len(Y)) for Y in y])

        if dim_x == 2 and dim_y == 2:
            for (X, Y, i) in zip(x, y, np.arange(len(x))):
                self.lines_x.append(np.array(X))
                self.lines_y.append(np.array(Y))
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
                    args_auxi["color"] = l_colors[i % len(l_colors)]
                if "label" in args and args_auxi["label"] == "":
                    del args_auxi["label"]
                    # Delete empty legend to prevent a warning message and the addition of an empty gray square
                self.param_lines.append(args_auxi)
        elif dim_y == 2:
            for (Y, i) in zip(y, np.arange(len(y))):
                self.lines_x.append(np.array(x))
                self.lines_y.append(np.array(Y))
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
                    args_auxi["color"] = l_colors[i % len(l_colors)]
                if "label" in args and args_auxi["label"] == "":
                    del args_auxi["label"]
                    # Delete empty legend to prevent a warning message and the addition of an empty gray square
                self.param_lines.append(args_auxi)
        elif len(y) != len(x):
            raise (UserWarning(
                "Graphique.line : the ordinate list should have the same size than the abscissa list len(x ): "
                + str(len(x)) + " len(y) : " + str(len(y))))
        else:
            self.lines_x.append(np.array(x))
            self.lines_y.append(np.array(y))
            self.err_y.append([])
            if marker != "" and not ("linestyle" in args):
                args["linestyle"] = ""
                args["marker"] = marker
            elif marker != "":
                args["marker"] = marker
            if "color" not in args:
                args["color"] = l_colors[(len(self.lines_x) - 1) % len(l_colors)]
            if "label" in args and args["label"] == "":
                del args["label"]
            self.param_lines.append(args)

    def text(self, x: list | np.ndarray, y: list | np.ndarray,
             s: list | np.ndarray, **args) -> None:
        """
        Equivalent to plt.text with a small improvement:
        if y is in two dimensions, the second dimension is plotted :
            self.line(x,[y1,y2], *args) is equivalent to
            plt.plot(x, y1; *args)
            plt.plot(x, y2, *args)
            if y1 and y2 have not the same size:
            self.line([x1,x2],[y1, y2], *args)
        :param x: Abscissa(s)
        :param y: Ordinate(s)
        :param s: Texts to plot
        :param args: Additional argument to plot() function like linestyle, color....
        :return: None
        """
        if type(s) is str:
            s: np.ndarray = np.array([s for X in x])
        if (len(np.shape(np.array(x))) == 2 and len(np.shape(np.array(y))) == 2 and
                np.shape(np.array(x))[1] == np.shape(np.array(y))[1]):
            for (X, Y, S) in zip(x, y, s):
                self.lines_t_x.append(X)
                self.lines_t_y.append(Y)
                self.lines_t_s.append(S)
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
                        args_auxi[i]["color"] = l_colors[(len(self.lines_x) + i - 1) % len(l_colors)]
                self.param_texts.extend(args_auxi)
        elif len(y) != len(x):
            raise ValueError("Graphique.text : the ordinate list should have the same size than "
                             "the abscissa list len(x): "
                             + str(len(x)) + " y : " + str(np.shape(np.array(y))))
        elif len(y) != len(s):
            raise ValueError("Graphique.text : the ordinate list should have the same size than "
                             "the text list len(s): "
                             + str(len(s)) + " y : " + str(np.shape(np.array(y))))
        else:
            self.lines_t_x.append(x)
            self.lines_t_y.append(y)
            self.lines_t_s.append(s)
            if "color" not in args:
                # args["color"] = 'C' + str(len(self.lines_x) % 10)
                args["color"] = l_colors[(len(self.lines_x) - 1) % len(l_colors)]
            self.param_texts.append(args)

    def point(self, xp: float | np.double, yp: float | np.double, marker: str = "o", **args) -> None:
        """
        Equivalent to plt.plot([X],[y],**args)
        :param xp: Abscissa(s)
        :param yp: Ordinate(s)
        :param marker: The marker (default="" (no marker), ex ".", ",", "o", "v"...
         see matplotlib documentation for all the possibility
        :param args: Additional argument to plot() function like linestyle, color....
        :return: None"""
        self.line([xp], [yp], marker=marker, **args)

    def errorbar(
            self, x: list | np.ndarray, y: list | np.ndarray, err_y: list | np.ndarray,
            marker: str = "", scale: str = "", **args: dict) -> None:
        """
        Equivalent to plt.errorbar
        :param x: Abscissa
        :param y: Ordinate
        :param err_y: Error associated with y
        :param marker: The marker (default="" (no marker), ex ".", ",", "o", "v"...
         see matplotlib documentation for all the possibility
        :param scale: The scales of (x, y) axis : default : "" (linear scale for both x and y
            - polar : polar projection : X=R and Y=Theta
            - loglog, logx, logy : Logarithmic scale for both, x or y axis
            - symloglog, symlogx, symlogy : Logarithmic scale for both, x or y axis with positive and négative values
        :param args: Additional argument to plot() function like linestyle, color....
        :return: None
        """
        if type(err_y) is str:
            raise TypeError("Attention, une liste d'erreur est nécessaire pour l'ajout des affichage_incertitudes "
                            "dans un Graphique")
        if len(err_y) != len(x):
            raise ValueError(
                "Attention, la liste des erreur doit être de la même taille que la liste des abscisse et"
                " celle des ordonnées : x :" +
                str(len(x)) + " y : " + str(len(y)) + " err y : " + str(len(err_y)))
        if scale == "":
            self.line(x, y, marker=marker, **args)
        elif scale == "polaire":
            self.polar(x, y, marker=marker, **args)
        elif scale == "loglog":
            self.loglog(x, y, marker=marker, **args)
        elif scale == "logx":
            self.logx(x, y, marker=marker, **args)
        elif scale == "logy":
            self.logy(x, y, marker=marker, **args)
        elif scale == "symloglog":
            self.loglog(x, y, marker=marker, **args)
        elif scale == "symlogx":
            self.logx(x, y, marker=marker, **args)
        elif scale == "symlogy":
            self.logy(x, y, marker=marker, **args)
        else:
            raise (ValueError("The scale " + scale + """ is not awailible. Please use : "", "polar",
            "loglog", "logx", "logy", "symloglog", "symlogx", "symlogy" """))

        self.err_y[-1] = err_y

    def errorplot(
            self, x: list | np.ndarray, y: list | np.ndarray, err_y: list | np.ndarray,
            marker: str = "", scale: str = "", **args: dict) -> None:
        """
        Equivalent to plt.errorbar but the error are not represented by errorbars but by a uniform
        colored polygon
        :param x: Abscissa
        :param y: Ordinate
        :param err_y: Error associated with y
        :param marker: The marker (default="" (no marker), ex ".", ",", "o", "v"...
         see matplotlib documentation for all the possibility
        :param scale: The scales of (x, y) axis : default : "" (linear scale for both x and y
            - polar : polar projection : X=R and Y=Theta
            - loglog, logx, logy : Logarithmic scale for both, x or y axis
            - symloglog, symlogx, symlogy : Logarithmic scale for both, x or y axis with positive and négative values
        :param args: Additional argument to plot() function like linestyle, color....
        :return: None
        """

        if type(err_y) is str:
            raise TypeError("Attention, une liste d'erreur est nécessaire pour l'ajout des affichage_incertitudes "
                            "dans un Graphique")
        if len(err_y) != len(x):
            raise ValueError(
                "Attention, la liste des erreur doit être de la même taille que la liste des abscisse et"
                " celle des ordonnées : x :" +
                str(len(x)) + " y : " + str(len(y)) + " err y : " + str(len(err_y)))
        if scale == "":
            self.line(x, y, marker=marker, **args)
        elif scale == "polaire":
            self.polar(x, y, marker=marker, **args)
        elif scale == "loglog":
            self.loglog(x, y, marker=marker, **args)
        elif scale == "logx":
            self.logx(x, y, marker=marker, **args)
        elif scale == "logy":
            self.logy(x, y, marker=marker, **args)
        elif scale == "symloglog":
            self.loglog(x, y, marker=marker, **args)
        elif scale == "symlogx":
            self.logx(x, y, marker=marker, **args)
        elif scale == "symlogy":
            self.logy(x, y, marker=marker, **args)
        else:
            raise (ValueError("The scale " + scale + """ is not awailible. Please use : "", "polar",
            "loglog", "logx", "logy", "symloglog", "symlogx", "symlogy" """))

        erry: list = list(y + err_y)
        erry2: list = list(y - err_y)
        erry2.reverse()
        erry.extend(erry2)
        x: list = list(x)
        x2: list = x.copy()
        x2.reverse()
        x.extend(x2)
        ind: np.ndarray = np.array([x, erry]).T
        self.polygon(ind, facecolor=self.param_lines[-1]["color"])

    def polar(self, R: list | np.ndarray, Theta: list | np.ndarray,
              marker: str | list = "", **args) -> None:
        """
        Equivalent to self.line in polar projection
        Attention: The order is opposit to the matplotlib one :
        The first argument is the radius, then the angle
        :param R: Radius
        :param Theta: Angle(s)
        :param marker: The marker (default="" (no marker), ex ".", ",", "o", "v"...
         see matplotlib documentation for all the possibility
        :param args: Additional argument to plot() function like linestyle, color....
        :return: None
        """
        self.line(R, Theta, marker, **args)
        self.config_ax(projection="polar")

    def loglog(self, x: np.ndarray | list, y: np.ndarray | list = None,
               marker: str | list = "", **args) -> None:
        """
        Equivalent to self.line with a logarithmique scale for both x and y-axis:
        if y is in two dimensions, the second dimension is plotted :
            self.line(x,[y1,y2], *args) is equivalent to
            plt.plot(x, y1, *args)
            plt.plot(x, y2, *args)
            if y1 and y2 have not the same size:
            self.line([x1,x2],[y1, y2], *args)
            If others arguments are list of the same size of x and y, they are also split :
            self.line((x1, x2], [y1, y2], marker=".", label=["Curve1", "Curve2"] is equivalent to
            plt.plot(x, y1, marker=".", label="Curve1")
            plt.plot(x, y2, marker=".", label="Curve2")
        :param x: Abscissa(s)
        :param y: Ordinate(s)
        :param marker: The marker (default="" (no marker), ex ".", ",", "o", "v"...
         see matplotlib documentation for all the possibility
        :param args: Additional argument to plot() function like linestyle, color....
        :return: None
        """
        self.line(x, y, marker, **args)
        self.config_ax(xscale="log", yscale="log")

    def symloglog(self, x: np.ndarray | list, y: np.ndarray | list = None,
                  marker: str | list = "", **args) -> None:
        """
        Equivalent to self.line with a logarithmique scale for both x and y axis
        The negatives values are also plotted:
        if y is in two dimensions, the second dimension is plotted :
            self.line(x,[y1,y2], *args) is equivalent to
            plt.plot(x, y1, *args)
            plt.plot(x, y2, *args)
            if y1 and y2 have not the same size:
            self.line([x1,x2],[y1, y2], *args)
            If others arguments are list of the same size of x and y, they are also split :
            self.line((x1, x2], [y1, y2], marker=".", label=["Curve1", "Curve2"] is equivalent to
            plt.plot(x, y1, marker=".", label="Curve1")
            plt.plot(x, y2, marker=".", label="Curve2")
        :param x: Abscissa(s)
        :param y: Ordinate(s)
        :param marker: The marker (default="" (no marker), ex ".", ",", "o", "v"...
         see matplotlib documentation for all the possibility
        :param args: Additional argument to plot() function like linestyle, color....
        :return: None
        """
        self.line(x, y, marker, **args)
        self.config_ax(xscale="symlog", yscale="symlog")

    def logx(self, x: np.ndarray | list, y: np.ndarray | list = None,
               marker: str | list = "", **args) -> None:
        """
        Equivalent to self.line with a logarithmique scale for x-axis:
        if y is in two dimensions, the second dimension is plotted :
            self.line(x,[y1,y2], *args) is equivalent to
            plt.plot(x, y1, *args)
            plt.plot(x, y2, *args)
            if y1 and y2 have not the same size:
            self.line([x1,x2],[y1, y2], *args)
            If others arguments are list of the same size of x and y, they are also split :
            self.line((x1, x2], [y1, y2], marker=".", label=["Curve1", "Curve2"] is equivalent to
            plt.plot(x, y1, marker=".", label="Curve1")
            plt.plot(x, y2, marker=".", label="Curve2")
        :param x: Abscissa(s)
        :param y: Ordinate(s)
        :param marker: The marker (default="" (no marker), ex ".", ",", "o", "v"...
         see matplotlib documentation for all the possibility
        :param args: Additional argument to plot() function like linestyle, color....
        :return: None
        """

    def symlogx(self, x: np.ndarray | list, y: np.ndarray | list = None,
                marker: str | list = "", **args) -> None:
        """
        Equivalent to self.line with a logarithmique scale for both x-axis
        The negatives values are also plotted:
        if y is in two dimensions, the second dimension is plotted :
            self.line(x,[y1,y2], *args) is equivalent to
            plt.plot(x, y1, *args)
            plt.plot(x, y2, *args)
            if y1 and y2 have not the same size:
            self.line([x1,x2],[y1, y2], *args)
            If others arguments are list of the same size of x and y, they are also split :
            self.line((x1, x2], [y1, y2], marker=".", label=["Curve1", "Curve2"] is equivalent to
            plt.plot(x, y1, marker=".", label="Curve1")
            plt.plot(x, y2, marker=".", label="Curve2")
        :param x: Abscissa(s)
        :param y: Ordinate(s)
        :param marker: The marker (default="" (no marker), ex ".", ",", "o", "v"...
         see matplotlib documentation for all the possibility
        :param args: Additional argument to plot() function like linestyle, color....
        :return: None
        """
        self.line(x, y, marker, **args)
        self.config_ax(xscale="symlog")

    def logy(self, x: np.ndarray | list, y: np.ndarray | list = None,
               marker: str | list = "", **args) -> None:
        """
        Equivalent to self.line with a logarithmique scale for y-axis:
        if y is in two dimensions, the second dimension is plotted :
            self.line(x,[y1,y2], *args) is equivalent to
            plt.plot(x, y1, *args)
            plt.plot(x, y2, *args)
            if y1 and y2 have not the same size:
            self.line([x1,x2],[y1, y2], *args)
            If others arguments are list of the same size of x and y, they are also split :
            self.line((x1, x2], [y1, y2], marker=".", label=["Curve1", "Curve2"] is equivalent to
            plt.plot(x, y1, marker=".", label="Curve1")
            plt.plot(x, y2, marker=".", label="Curve2")
        :param x: Abscissa(s)
        :param y: Ordinate(s)
        :param marker: The marker (default="" (no marker), ex ".", ",", "o", "v"...
         see matplotlib documentation for all the possibility
        :param args: Additional argument to plot() function like linestyle, color....
        :return: None
        """

    def symlogy(self, x: np.ndarray | list, y: np.ndarray | list = None,
                  marker: str | list = "", **args) -> None:
        """
        Equivalent to self.line with a logarithmique scale y-axis
        The negatives values are also plotted:
        if y is in two dimensions, the second dimension is plotted :
            self.line(x,[y1,y2], *args) is equivalent to
            plt.plot(x, y1, *args)
            plt.plot(x, y2, *args)
            if y1 and y2 have not the same size:
            self.line([x1,x2],[y1, y2], *args)
            If others arguments are list of the same size of x and y, they are also split :
            self.line((x1, x2], [y1, y2], marker=".", label=["Curve1", "Curve2"] is equivalent to
            plt.plot(x, y1, marker=".", label="Curve1")
            plt.plot(x, y2, marker=".", label="Curve2")
        :param x: Abscissa(s)
        :param y: Ordinate(s)
        :param marker: The marker (default="" (no marker), ex ".", ",", "o", "v"...
         see matplotlib documentation for all the possibility
        :param args: Additional argument to plot() function like linestyle, color....
        :return: None
        """
        self.line(x, y, marker, **args)
        self.config_ax(yscale="symlog")

    def histogram(
            self, values: np.ndarray, weights: np.ndarray | None = None,
            normalization: bool = True, statistic: str = 'sum', bins: int = 10, **args) -> None:
        """
        Plot the histogram of values
        :param values: The values to histogramed
        :param weights: The weights to be applied to values (default one)
        :param normalization: If the histogram is normalized or not
        :param statistic: The statistic to compute (default is 'sum'). The following statistics are available:
'mean': compute the mean of values for points within each bin. Empty bins will be represented by NaN.
'std': compute the standard deviation within each bin. This is implicitly calculated with ddof=0.
'median': compute the median of values for points within each bin. Empty bins will be represented by NaN.
'count': compute the count of points within each bin. This is identical to an unweighted histogram. values array is not referenced.
'sum': compute the sum of values for points within each bin. This is identical to a weighted histogram.
'min': compute the minimum of values for points within each bin. Empty bins will be represented by NaN.
'max': compute the maximum of values for point within each bin. Empty bins will be represented by NaN.
function : a user-defined function which takes a 1D array of values, and outputs a single numerical statistic. This function will be called on the values in each bin. Empty bins will be represented by function([]), or NaN if this returns an error.
        :param bins: Number of bins in the histogram
        :param args: Additionals argument for `sp.binned_statistic`
        :return: None
        """

        if weights is None:
            weights = np.ones(len(values))
        vals, bds, indices = sp.binned_statistic(
            values, weights, statistic, bins, **args)

        if normalization:
            vals /= len(values)
            vals /= bds[1:] - bds[:-1]
        self.bords_histogramme.append(bds)
        self.vals_histogramme.append(vals)
        self.param_histogrammes.append(args)

    def image(self, array_image: np.ndarray,
              x_axe: list | np.ndarray, y_axe: list | np.ndarray, **args) -> None:
        """
        Plot the array image through plt.pcolor
        :param array_image: The matrix (2D) to be plotted
        :param x_axe: the x-axes coordinate (for the array)
        :param y_axe: the y-axes coordinate (for the array)
        :param args: Additionals arguments for pcolor
        :return:
        """
        self.array_image = array_image
        self.x_axe_image = np.array(x_axe)
        self.y_axe_image = np.array(y_axe)
        self.param_image = args

    def contours(
            self, levels: np.ndarray | list | None = None, array_contours: np.ndarray | None = None,
            x_axe: list | np.ndarray | None = None,
            y_axe: list | np.ndarray | None = None, labels: list | np.ndarray | None = None,
            labels_mask: np.ndarray | None = None, **args):
        """
        Plot the level lines associated to self.array_image or array_contours
        :param levels: Nombre (or list of) levels to plot
        :param array_contours: If not None, the reference array to determine the level
         (default, array_contours=self.array_image)
        :param x_axe: the x-axes coordinate (for the array if array_contour is not None)
        :param y_axe: the y-axes coordinate (for the array if array_contour is not None)
        :param labels: the labels of each level line
        :param labels_mask: the mask of levels line to show the labels
        :param args: additional arguments
        :return: None
        """
        idx_levels: np.ndarray | None = None
        if type(levels) is list or type(levels) is np.ndarray:
            idx_levels = np.argsort(levels)
            levels = levels[idx_levels]

        if "colors" in args.keys() and (type(args["colors"]) is list
                                        or type(args["colors"]) is np.ndarray):
            self.color_label_contours = args["colors"]
            del args["colors"]

        if levels is not None:
            args['levels'] = levels
            if labels is not None:
                if len(labels) != len(levels):
                    raise UserWarning("Graphique.contours : the labels size should be equal to the levels size: levels",
                                      len(levels), "labels :", len(labels))
                self.clabels = labels[idx_levels]

                if labels_mask is not None:
                    if len(labels_mask) != len(levels):
                        raise UserWarning("Graphique.contours : the labels_mask size should be equal "
                                          "to the levels/labels size: levels",
                                          len(levels), "labels_mask :", len(labels_mask))
                    self.clabels_mask = labels_mask[idx_levels]
        if array_contours is None:
            self.tab_contours_is_image = True
            if type(args['levels']) is int:
                self.nb_contours = args['levels']
                del args['levels']
            else:
                self.nb_contours = len(args['levels'])
                self.levels = args['levels']
                del args['levels']
            liste_intersect: list[str] = ['alpha0', 'vmin', 'vmax', 'norm']
            if "colors" not in args:
                liste_intersect.append("cmap")
            for p in liste_intersect:
                if p in self.param_image:
                    self.param_contours[p] = self.param_image[p]
            self.param_contours.update(args)
        else:
            self.array_contours = array_contours
            self.tab_contours_is_image = False
            self.x_axe_contours = x_axe
            self.y_axe_contours = y_axe
            self.param_contours = args

    def polygon(self, ind, alpha: float | np.double = 0.7, facecolor: str = 'C3', **args) -> None:
        """
        Plot a uniformly colored polygon
        :param ind: 2-dimensional array/list of the coordinate of the polygon characteristics points
        ind[:, 0] point's abscissas
        ind[:, 1] point's ordinate
        :param alpha:
        :param facecolor:
        :param args:
        :return:
        """
        self.index_polygons.append(ind)
        args["alpha0"] = alpha
        args['facecolor'] = facecolor
        self.param_polygons.append(args)
        args = args.copy()
        args['color'] = facecolor
        del args['facecolor']
        if "label" in args:
            del args["label"]
        self.line(ind[:, 0], ind[:, 1], **args)  # border of the polygon

    def config_ax(self, **dico) -> None:
        """
        Additionals configurations for ax
        :param dico:
        Keywords awalible (see matplotlib documentation)

            - sharex, shareyAxes, optional :The x- or y-axis is shared with the x- or y-axis in the input Axes. Note that it is not possible to unshare axes.

            - frameonbool, default: True : Whether the Axes frame is visible.

            - box_aspectfloat, optional : Set a fixed aspect for the Axes box,
             i.e. the ratio of height to width. See set_box_aspect for details.

            - forward_navigation_eventsbool or "auto", default: "auto"
            Control whether pan/zoom events are passed through to Axes below this one. "auto" is True for axes with an
             invisible patch and False otherwise.

            - Other optional keyword arguments:

            -- adjustable {'box', 'datalim'}

            -- agg_filter : a filter function, which takes a (m, n, 3) float array and a dpi value, and returns
             a (m, n, 3) array and two offsets from the bottom left corner of the image

            -- alpha : scalar or None

            -- anchor : (float, float) or {'C', 'SW', 'S', 'SE', 'E', 'NE', ...}

            -- animated : bool

            -- aspect : {'auto', 'equal'} or float

            -- autoscale_on : bool

            -- autoscalex_on

            -- autoscaley_on

            -- axes_locator : Callable[[Axes, Renderer], Bbox]

            -- axisbelow : bool or 'line'

            -- box_aspect : float or None

            -- clip_on : bool

            -- facecolor or fc : color

            -- figure : Figure

            -- forward_navigation_events : bool or "auto"

            -- frame_on : bool

            -- gid : str

            -- in_layout : bool

            -- label : object

            -- mouseover : bool

            -- navigate : bool

            -- navigate_mode

            -- picker : None or bool or float

            -- position : [left, bottom, width, height]

            -- rasterization_zorder : float or None

            -- rasterized : bool

            -- sketch_params : (scale: float, length: float, randomness: float)

            -- snap : bool or None

            -- subplotspec

            -- title : str

            -- url : str

            -- visible : bool

            -- xbound : (lower: float, upper: float)

            -- xlabel : str

            -- xlim : (left: float, right: float)

            -- xmargin : float greater than -0.5

            -- xscale

            -- xticklabels

            -- xticks

            -- ybound : (lower: float, upper: float)

            -- ylabel : str

            -- ylim : (bottom: float, top: float)

            -- ymargin : float greater than -0.5

            -- yscale

            -- yticklabels

            -- yticks

            -- zorder : float
        :return: None
        """
        if 'xticks' in dico:
            self.x_axe = dico['xticks']
            del dico['xticks']
        if 'yticks' in dico:
            self.y_axe = dico['yticks']
            del dico['yticks']
        if 'xticklabels' in dico:
            self.labels_x_ticks = dico['xticklabels']
            del dico['xticklabels']
        if "yticklabels" in dico:
            self.labels_y_ticks = dico['yticklabels']
            del dico['yticklabels']
        if "Figure" in dico:
            self.fig = dico["Figure"]
            del dico["Figure"]
        self.param_ax.update(dico)

    def config_legende(self, **dico) -> None:
        """
        :param dico: additionals parameters for the legend
        (see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html)

        - locstr  default: rcParams["legend.loc"] (default: 'best')  The location of the legend.
                The strings 'upper left', 'upper right', 'lower left', 'lower right' place the legend at
                 the corresponding corner of the axes.
                The strings 'upper center', 'lower center', 'center left', 'center right' place the
                 legend at the center of the corresponding edge of the axes.
                The string 'center' places the legend at the center of the axes.
                The string 'best' places the legend at the location, among the nine locations defined so far, with the
                 minimum overlap with other drawn artists. This option can be quite slow for plots with large amounts
                 of data; your plotting speed may benefit from providing a specific location.
                The location can also be a 2-tuple giving the coordinates of the lower-left corner of the legend in
                axes coordinates (in which case bbox_to_anchor will be ignored).
                For back-compatibility, 'center right' (but no other location) can also be spelled 'right', and each
                 "string" location can also be given as a numeric value:

        - bbox_to_anchorBboxBase, 2-tuple, or 4-tuple of floats
            Box that is used to position the legend in conjunction with loc.
            Defaults to axes.bbox (if called as a method to Axes.legend) or figure.bbox (if Figure.legend).
            This argument allows arbitrary placement of the legend.
            Bbox coordinates are interpreted in the coordinate system given by bbox_transform, with the default
            transform Axes or Figure coordinates, depending on which legend is called.
            If a 4-tuple or BboxBase is given, then it specifies the bbox (x, y, width, height) that the legend is
            placed in. To put the legend in the best location in the bottom right quadrant of the Axes (or figure):

        - loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5)
        A 2-tuple (x, y) places the corner of the legend specified by loc at x, y. For example,
         to put the legend's upper right-hand corner in the center of the Axes (or figure)
         the following keywords can be used: loc='upper right', bbox_to_anchor=(0.5, 0.5)

        - ncolsint, default: 1 : The number of columns that the legend has.
        For backward compatibility, the spelling ncol is also supported but it is discouraged.
         If both are given, ncols takes precedence.


        - fontsize int or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
            The font size of the legend. If the value is numeric the size will be the absolute font size in points. String values are relative to the current default font size. This argument is only used if prop is not specified.

        - labelcolorstr or list, default: rcParams["legend.labelcolor"] (default: 'None')
            The color of the text in the legend. Either a valid color string (for example, 'red'),
             or a list of color strings. The labelcolor can also be made to match the color of the line or marker using
              'linecolor', 'markerfacecolor' (or 'mfc'), or 'markeredgecolor' (or 'mec').
            Labelcolor can be set globally using rcParams["legend.labelcolor"] (default: 'None').
             If None, use rcParams["text.color"] (default: 'black').

        - numpointsint, default: rcParams["legend.numpoints"] (default: 1)
            The number of marker points in the legend when creating a legend entry for a Line2D (line).

        - scatterpointsint, default: rcParams["legend.scatterpoints"] (default: 1)
            The number of marker points in the legend when creating a legend entry for a PathCollection (scatter plot).

        - scatteryoffsets : iterable of floats, default: [0.375, 0.5, 0.3125]
            The vertical offset (relative to the font size) for the markers created for a scatter plot legend entry.
            0.0 is at the base the legend text, and 1.0 is at the top. To draw all markers at the same height,
            set to [0.5].

        - markerscalefloat, default: rcParams["legend.markerscale"] (default: 1.0)
        The relative size of legend markers compared to the originally drawn ones.

        - markerfirstbool, default: True If True, legend marker is placed to the left of the legend label.
        If False, legend marker is placed to the right of the legend label.

        - reversebool, default: False  If True, the legend labels are displayed in reverse order from the input.
         If False, the legend labels are displayed in the same order as the input.
                Added in version 3.7.

        - frameonbool, default: rcParams["legend.frameon"] (default: True)
            Whether the legend should be drawn on a patch (frame).

        - fancyboxbool, default: rcParams["legend.fancybox"] (default: True)
            Whether round edges should be enabled around the FancyBboxPatch which makes up the legend's background.

        - shadowNone, bool or dict, default: rcParams["legend.shadow"] (default: False)
            Whether to draw a shadow behind the legend. The shadow can be configured using Patch keywords.
            Customization via rcParams["legend.shadow"] (default: False) is currently not supported.

        - framealpha float, default: rcParams["legend.framealpha"] (default: 0.8)
            The alpha transparency of the legend's background. If shadow is activated and framealpha is None,
             the default value is ignored.

        - facecolor "inherit" or color, default: rcParams["legend.facecolor"] (default: 'inherit')
            The legend's background color. If "inherit", use rcParams["axes.facecolor"] (default: 'white').

        - edgecolor "inherit" or color, default: rcParams["legend.edgecolor"] (default: '0.8')
            The legend's background patch edge color. If "inherit", use rcParams["axes.edgecolor"] (default: 'black').

        - mode : {"expand", None}
            If mode is set to "expand" the legend will be horizontally expanded to fill the Axes area
             (or bbox_to_anchor if defines the legend's size).

        - bbox_transformNone or Transform  The transform for the bounding box (bbox_to_anchor).
        For a value of None (default) the Axes' transAxes transform will be used.

        - titlestr or None :  The legend's title. Default is no title (None).

        - title_fontproperties : None or FontProperties or dict
                The font properties of the legend's title. If None (default), the title_fontsize argument will be used
                 if present; if title_fontsize is also None, the current rcParams["legend.title_fontsize"]
                  (default: None) will be used.

        - title_fontsize int or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'},
            default: rcParams["legend.title_fontsize"] (default: None)
            The font size of the legend's title. Note: This cannot be combined with title_fontproperties.
             If you want to set the fontsize alongside other font properties, use the size parameter
              in title_fontproperties.

        - alignment{'center', 'left', 'right'}, default: 'center'
            The alignment of the legend title and the box of entries. The entries are aligned as a single block,
            so that markers always lined up.

        - borderpad float, default: rcParams["legend.borderpad"] (default: 0.4)
            The fractional whitespace inside the legend border, in font-size units.

        - labelspacing  float, default: rcParams["legend.labelspacing"] (default: 0.5)
            The vertical space between the legend entries, in font-size units.

        - handlelength float, default: rcParams["legend.handlelength"] (default: 2.0)
            The length of the legend handles, in font-size units.

        - handleheight float, default: rcParams["legend.handleheight"] (default: 0.7)
            The height of the legend handles, in font-size units.

        - handletextpad float, default: rcParams["legend.handletextpad"] (default: 0.8)
            The pad between the legend handle and text, in font-size units.

        - borderaxespad float, default: rcParams["legend.borderaxespad"] (default: 0.5)
            The pad between the Axes and legend border, in font-size units.

        - columnspacing float, default: rcParams["legend.columnspacing"] (default: 2.0)
            The spacing between columns, in font-size units.

        - draggablebool, default: False
            Whether the legend can be dragged with the mouse.
        :return: None
        """
        self.param_legende.update(dico)

    def config_labels_contours(self, **dico) -> None:
        """
        Additionals configurations for the contours labels
        see : https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.clabel.html
        :param dico:
            - fontsize str or float, default: rcParams["font.size"] (default: 10.0)
                Size in points or relative size e.g., 'smaller', 'x-large'.
                See plt.Text.set_size for accepted string values.

            - colorscolor or colors or None, default: None
                The label colors:

                    If None, the color of each label matches the color of the corresponding contour.

                    If one string color, e.g., colors = 'r' or colors = 'red', all labels will be plotted in this color.

                    If a tuple of colors (string, float, RGB, etc), different labels will be plotted in different
                     colors in the order specified.

            - inlinebool, default: True
                If True the underlying contour is removed where the label is placed.

            - inline_spacingfloat, default: 5
                Space in pixels to leave on each side of label when placing inline.
                This spacing will be exact for labels at locations where the contour is straight, less so for labels on curved contours.

            - fmt str, optional
                How the levels are formatted:  it is interpreted as a %-style format string.
                The default is to use a standard ScalarFormatter.

            - manual bool or iterable, default: False
                If True, contour labels will be placed manually using mouse clicks. Click the first button near a
                 contour to add a label, click the second button (or potentially both mouse buttons at once) to finish
                  adding labels. The third button can be used to remove the last label added, but only if labels are not
                   inline. Alternatively, the keyboard can be used to select label locations (enter to end label
                   placement, delete or backspace act like the third mouse button, and any other key will select a label
                    location).
                manual can also be an iterable object of (x, y) tuples. Contour labels will be created as if mouse is clicked at each (x, y) position.

            - rightside_upbool, default: True
                If True, label rotations will always be plus or minus 90 degrees from level.

            - use_clabeltext bool, default: False
                If True, use Text.set_transform_rotates_text to ensure that label rotation is updated whenever the Axes aspect changes.

            - zorder float or None, default: (2 + contour.get_zorder())
                zorder of the contour labels.

        :return: None
        """
        self.param_labels_contours.update(dico)

    def config_fig(self, **dico) -> None:
        """
        Additionnals parameters to configure the Figure
        :param dico:

        - igsize2-tuple of floats, default: rcParams["figure.figsize"] (default: [6.4, 4.8])
            Figure dimension (width, height) in inches.
        - dpi float, default: rcParams["figure.dpi"] (default: 100.0)
            Dots per inch.
        - facecolor default: rcParams["figure.facecolor"] (default: 'white')
            The figure patch facecolor.
        - edgecolor default: rcParams["figure.edgecolor"] (default: 'white')
            The figure patch edge color.
        - linewidthfloat
            The linewidth of the frame (i.e. the edge linewidth of the figure patch).

        - frameonbool, default: rcParams["figure.frameon"] (default: True)
            If False, suppress drawing the figure background patch.

        - layout {'constrained', 'compressed', 'tight', 'none', LayoutEngine, None}, default: None

            The layout mechanism for positioning of plot elements to avoid overlapping Axes decorations (labels, ticks, etc). Note that layout managers can have significant performance penalties.

                'constrained': The constrained layout solver adjusts Axes sizes to avoid overlapping Axes decorations. Can handle complex plot layouts and colorbars, and is thus recommended.

                See Constrained layout guide for examples.

                'compressed': uses the same algorithm as 'constrained', but removes extra space between fixed-aspect-ratio Axes. Best for simple grids of Axes.

                'tight': Use the tight layout mechanism. This is a relatively simple algorithm that adjusts the subplot parameters so that decorations do not overlap.

                See Tight layout guide for examples.

                'none': Do not use a layout engine.

                A LayoutEngine instance. Builtin layout classes are ConstrainedLayoutEngine and TightLayoutEngine, more easily accessible by 'constrained' and 'tight'. Passing an instance allows third parties to provide their own layout engine.

            If not given, fall back to using the parameters tight_layout and constrained_layout, including their config defaults rcParams["figure.autolayout"] (default: False) and rcParams["figure.constrained_layout.use"] (default: False).

        - alpha scalar or None

        - animated bool

        - clip_on bool

        - constrained_layout unknown

        - constrained_layout_pads unknown

        - dpi float

        - edgecolor color

        - facecolor color

        - figheight float

        - figwidth float

        - frameon bool

        - gid str

        - in_layout bool

        - layout_engine  {'constrained', 'compressed', 'tight', 'none', LayoutEngine, None}

        - linewidth number

        - mouseover bool

        - picker None or bool or float or callable

        - rasterized bool

        - size_inches  (float, float) or float

        - sketch_params (scale: float, length: float, randomness: float)

        - snap bool or None

        - tight_layout

        - url str

        - visible bool

        - zorder float

        :return:
        """
        self.param_fig.update(dico)

    def config_enrg_fig(self, **dico) -> None:
        """
        Additionals parameters for the Figure saving
    see https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.savefig.html#matplotlib.figure.Figure.savefig
        :param dico:
            - igsize 2-tuple of floats, default: rcParams["figure.figsize"] (default: [6.4, 4.8])
                Figure dimension (width, height) in inches.

            - dpi float, default: rcParams["figure.dpi"] (default: 100.0)
                Dots per inch.

            - facecolor default: rcParams["figure.facecolor"] (default: 'white')
                The figure patch facecolor.

            - edgecolor default: rcParams["figure.edgecolor"] (default: 'white')
                The figure patch edge color.

            - linewidth float
                The linewidth of the frame (i.e. the edge linewidth of the figure patch).

            - frameon bool, default: rcParams["figure.frameon"] (default: True)
                If False, suppress drawing the figure background patch.

            - layout {'onstrained', 'compressed', 'tight', 'none', LayoutEngine, None}, default: None

                The layout mechanism for positioning of plot elements to avoid overlapping Axes decorations
                 (labels, ticks, etc). Note that layout managers can have significant performance penalties.

                    'constrained': The constrained layout solver adjusts Axes sizes to avoid overlapping Axes
                     decorations. Can handle complex plot layouts and colorbars, and is thus recommended.

                    See Constrained layout guide for examples.

                    'compressed': uses the same algorithm as 'constrained', but removes extra space between
                    fixed-aspect-ratio Axes. Best for simple grids of Axes.

                    'tight': Use the tight layout mechanism. This is a relatively simple algorithm that adjusts the
                     subplot parameters so that decorations do not overlap.

                    See Tight layout guide for examples.

                    'none': Do not use a layout engine.

                    A LayoutEngine instance. Builtin layout classes are ConstrainedLayoutEngine and TightLayoutEngine,
                     more easily accessible by 'constrained' and 'tight'. Passing an instance allows third parties to
                      provide their own layout engine.

                If not given, fall back to using the parameters tight_layout and constrained_layout, including their
                 config defaults rcParams["figure.autolayout"] (default: False) and
                 rcParams["figure.constrained_layout.use"] (default: False).

            - alpha  scalar or None

            - animated  bool

            - clip_on bool

            - constrained_layout  unknown

            - constrained_layout_pads unknown

            - dpi float

            - edgecolor color

            - facecolor  color

            - figheight float

            - figwidth float

            - frameon bool

            - gid str

            - in_layout bool

            - label object

            - layout_engine {'constrained', 'compressed', 'tight', 'none', LayoutEngine, None}

            - linewidth number

            - mouseover bool

            - picker None or bool or float

            - rasterized bool

            - size_inches (float, float) or float

            - sketch_params (scale: float, length: float, randomness: float)

            - snap bool or None

            - tight_layout unknown

            - transform  Transform

            - url str

            - visible bool

            - zorder float
        :return: None
        """
        self.param_enrg_fig.update(dico)

    def config_font(self, **dico) -> None:
        """
        Global font parameter
        :param dico
            'family' : 'fantasy','monospace','sans-serif','serif','cursive'
            'styles' : 'normal', 'italic', 'oblique'
            'size' : valeur numérique
            'variants' : 'normal', 'small-caps'
            'weight' : 'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'
-       :return None
        """
        k: list[str] = dico.keys()
        vals: list = dico.values()
        dico: dict = {}
        for K, L in zip(k, vals):
            if "font." not in K:
                dico['font.' + K] = L
            else:
                dico[K] = L
        self.param_font.update(dico)

    def config_colorbar(self, index_colorbar: int = ii_max, ticks: list | np.ndarray | None = None,
                        **dico) -> None:
        """
        Colorbar additianal parameter
        :param index_colorbar:
            The index of the colorbar (default the parameters are added for all colorbars)
            0 is the index of the image's colorbar
        :param ticks: The colorbar's ticks. If None, ticks are determined automatically from the input.
        :param dico: the parameter dictionary
        :return:
        """
        if index_colorbar == ii_max:
            for d in self.param_colorbar:
                d.update(dico)
            if ticks is not None:
                self.ticks_colorbar = [ticks for t in self.ticks_colorbar]
        else:
            if ticks is not None:
                self.ticks_colorbar[index_colorbar] = ticks
            self.param_colorbar[index_colorbar].update(dico)

    def fond_noir(self) -> None:
        self.style = 'dark_background'
        for d in self.param_lines:
            if "color" in d.keys() and (isinstance(d["color"], str ) and d["color"] == "k"):
                d["color"] = "w"
        for d in self.param_contours:
            if "color" in d.keys() and (isinstance(d["color"], str ) and d["color"] == "k"):
                d["color"] = "w"
        for d in self.param_polygons:
            if "facecolor" in d.keys() and (isinstance(d["color"], str ) and d["color"] == "k"):
                d["facecolor"] = "w"

    def fond_blanc(self) -> None:
        self.style = 'default'
        for d in self.param_lines:
            if "color" in d.keys() and (isinstance(d["color"], str ) and d["color"] == "w"):
                d["color"] = "k"
        if "colors" in self.param_contours.keys():
            for i in range(len(self.param_contours["colors"])):
                if (isinstance(self.param_contours["colors"][i], str)
                        and self.param_contours["colors"][i] == "w"):
                    self.param_contours["colors"] = "k"
        for d in self.param_polygons:
            if "facecolor" in d.keys() and d["facecolor"] == "w":
                d["facecolor"] = "k"

    def projection_colorbar(self) -> None:
        """
        Build the custom colorbar if it exists
        :return: None
        """
        if self.custum_colorbar_colors is not None:
            for i in range(len(self.custum_colorbar_colors)):
                cmap = mpl.colors.ListedColormap(self.custum_colorbar_colors[i])
                norm = mpl.colors.BoundaryNorm(self.custum_colorbar_values[i], cmap.N)
                params: dict = self.param_colorbar[i + 1].copy()
                if len(self.ticks_colorbar[i + 1]) > 0:
                    params["ticks"] = self.ticks_colorbar[i + 1]
                self.colorbar = self.fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                                                  **params)

    def projection_lines(self) -> None:
        # Affiche les lines sur l'axe
        with mpl.rc_context(self.param_font):
            if self.ax is None:
                self.ax = self.fig.add_axes([0, 0, 1, 1], **self.param_ax)
            for i in range(len(self.lines_x)):
                if len(self.param_lines[i]) > 0:
                    if len(self.err_y[i]) > 0:
                        self.ax.errorbar(
                            self.lines_x[i], self.lines_y[i], self.err_y[i], **self.param_lines[i])
                    else:
                        self.ax.plot(
                            self.lines_x[i], self.lines_y[i], **self.param_lines[i])
                else:
                    self.ax.plot(self.lines_x[i], self.lines_y[i])
            if len(self.x_axe) == 0 or self.x_axe[0] > -np.inf:
                if len(self.x_axe) > 0:
                    self.ax.set_xlim([self.x_axe.min(), self.x_axe.max()])
                self.ax.set_xticks(self.x_axe)
            if len(self.labels_x_ticks) == 0 or self.labels_x_ticks[0] != "empty":
                self.ax.set_xticklabels(self.labels_x_ticks)
            if len(self.y_axe) == 0 or self.y_axe[0] > -np.inf:
                if len(self.y_axe) > 0:
                    self.ax.set_ylim([self.y_axe.min(), self.y_axe.max()])
                self.ax.set_yticks(self.y_axe)
            if len(self.labels_y_ticks) == 0 or self.labels_y_ticks[0] != "empty":
                self.ax.set_yticklabels(self.labels_y_ticks)
            if self.title != "":
                self.ax.set_title(self.title)

    def projection_texts(self) -> None:
        # Affiche les lines sur l'axe
        with mpl.rc_context(self.param_font):
            if self.ax is None:
                self.ax = self.fig.add_axes([0, 0, 1, 1], **self.param_ax)
            for i in range(len(self.lines_t_x)):
                for (X, Y, S) in zip(self.lines_t_x[i], self.lines_t_y[i], self.lines_t_s[i]):
                    self.ax.text(X, Y, S, **self.param_texts[i])

    def projection_histogrammes(self) -> None:
        with mpl.rc_context(self.param_font):
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
            if np.any([len(self.param_lines[i]) > 0 for i in range(len(self.param_lines))]):
                plt.legend(**self.param_legende)
            if len(self.x_axe) == 0 or self.x_axe[0] > -np.inf:
                if len(self.x_axe) > 0:
                    self.ax.set_xlim([self.x_axe.min(), self.x_axe.max()])
                self.ax.set_xticks(self.x_axe)
            if len(self.labels_x_ticks) == 0 or self.labels_x_ticks[0] != "empty":
                self.ax.set_xticklabels(self.labels_x_ticks)
            if len(self.y_axe) == 0 or self.y_axe[0] > -np.inf:
                if len(self.y_axe) > 0:
                    self.ax.set_ylim([self.y_axe.min(), self.y_axe.max()])
                self.ax.set_yticks(self.y_axe)
            if len(self.labels_y_ticks) == 0 or self.labels_y_ticks[0] != "empty":
                self.ax.set_yticklabels(self.labels_y_ticks)
            if self.title != "":
                self.ax.set_title(self.title)

    def projection_image(self) -> None:
        # Affiche l'image
        with mpl.rc_context(self.param_font):
            if self.ax is None:
                self.ax = self.fig.add_axes([0, 0, 1, 1], **self.param_ax)
            param_tableau: dict = self.param_image.copy()
            if ("scale" in self.param_colorbar.keys()
                    and self.param_colorbar["scale"] == "log"):
                if "vmin" in param_tableau.keys():
                    vmin = param_tableau["vmin"]
                    del param_tableau["vmin"]
                else:
                    vmin = self.array_image.min()
                if "vmax" in param_tableau.keys():
                    vmax = param_tableau["vmax"]
                    del param_tableau["vmax"]
                else:
                    vmax = self.array_image.max()
                if len(self.array_image.shape) == 2:
                    carte_xy = self.ax.pcolor(self.x_axe_image,
                                              self.y_axe_image, self.array_image,
                                              norm=LogNorm(vmin=vmin,
                                                           vmax=vmax),
                                              **param_tableau)
                else:
                    if "cmap" in param_tableau:
                        del param_tableau["cmap"]
                    if self.array_image.shape[2] == 3:
                        self.ax.pcolor(self.x_axe_image,
                                       self.y_axe_image, self.array_image[:, :, 0],
                                       norm=LogNorm(vmin=vmin, vmax=vmax),
                                       cmap="Reds", **param_tableau)
                        self.ax.pcolor(self.x_axe_image,
                                       self.y_axe_image, self.array_image[:, :, 1],
                                       norm=LogNorm(vmin=vmin, vmax=vmax),
                                       cmap="Greens", **param_tableau)
                        self.ax.pcolor(self.x_axe_image,
                                       self.y_axe_image, self.array_image[:, :, 2],
                                       norm=LogNorm(vmin=vmin, vmax=vmax),
                                       cmap="Blues", **param_tableau)
                    elif self.array_image.shape[1] == 3:
                        self.ax.pcolor(self.x_axe_image,
                                       self.y_axe_image, self.array_image[:, 0, :].T,
                                       norm=LogNorm(vmin=vmin, vmax=vmax), cmap="Reds",
                                       **param_tableau)
                        self.ax.pcolor(self.x_axe_image,
                                       self.y_axe_image, self.array_image[:, 1, :].T,
                                       norm=LogNorm(vmin=vmin, vmax=vmax),
                                       cmap="Greens", **param_tableau)
                        self.ax.pcolor(self.x_axe_image,
                                       self.y_axe_image, self.array_image[:, 2, :].T,
                                       norm=LogNorm(vmin=vmin, vmax=vmax),
                                       cmap="Blues", **param_tableau)
                    else:
                        self.ax.pcolor(self.x_axe_image,
                                       self.y_axe_image, self.array_image[0, :, :].T,
                                       norm=LogNorm(vmin=vmin, vmax=vmax), cmap="Reds",
                                       **param_tableau)
                        self.ax.pcolor(self.x_axe_image,
                                       self.y_axe_image, self.array_image[1, :, :].T,
                                       norm=LogNorm(vmin=vmin, vmax=vmax),
                                       cmap="Greens", **param_tableau)
                        self.ax.pcolor(self.x_axe_image,
                                       self.y_axe_image, self.array_image[2, :, :].T,
                                       norm=LogNorm(vmin=vmin, vmax=vmax),
                                       cmap="Blues", **param_tableau)
            else:
                if len(self.array_image.shape) == 2:
                    carte_xy = self.ax.pcolor(self.x_axe_image,
                                              self.y_axe_image, self.array_image,
                                              **param_tableau)
                else:

                    if "cmap" in param_tableau:
                        del param_tableau["cmap"]
                    self.ax.pcolor(self.x_axe_image,
                                   self.y_axe_image, self.array_image[:, :, 0],
                                   cmap="Reds", **param_tableau)
                    self.ax.pcolor(self.x_axe_image,
                                   self.y_axe_image, self.array_image[:, :, 1],
                                   cmap="Greens", **param_tableau)
                    self.ax.pcolor(self.x_axe_image,
                                   self.y_axe_image, self.array_image[:, :, 2],
                                   cmap="Blues", **param_tableau)
            params_cb: dict = self.param_colorbar[0].copy()
            if len(self.ticks_colorbar[0]) > 0:
                params_cb["ticks"] = self.ticks_colorbar[0]
            if 'scale' in params_cb.keys():
                del params_cb['scale']

            if "hide" in params_cb.keys() and not params_cb["hide"]:
                del params_cb["hide"]
            if 'hide' not in params_cb.keys():
                if self.fig is None:
                    self.colorbar = plt.colorbar(carte_xy,
                                                 **params_cb)
                    # self.fig.colorbar(carte_xy)
                else:
                    self.colorbar = plt.colorbar(carte_xy,
                                                 **params_cb)
            if self.title != "":
                self.ax.set_title(self.title)

    def projection_contours(self) -> None:
        params: dict = self.param_contours.copy()
        if len(self.color_label_contours) > 0:
            params["colors"] = self.color_label_contours
        with mpl.rc_context(self.param_font):
            if self.ax is None:
                self.ax = self.fig.add_axes([0, 0, 1, 1], **self.param_ax)
            if len(self.levels) > 0:
                levels = self.levels  # print("levels",levels)
            else:
                levels = self.nb_contours  # print("levels",levels)
            if self.tab_contours_is_image:
                if len(self.x_axe_image) != self.array_image.shape[1]:
                    x_axe = self.x_axe_image[:-1] + abs(
                        abs(self.x_axe_image[1:])
                        - abs(self.x_axe_image[:-1]))
                else:
                    x_axe = self.x_axe_image
                if len(self.y_axe_image) != self.array_image.shape[0]:
                    y_axe = self.y_axe_image[:-1] + abs(
                        abs(self.y_axe_image[1:])
                        - abs(self.y_axe_image[:-1]))
                else:
                    y_axe = self.y_axe_image

                cs = self.ax.contour(x_axe, y_axe,
                                     self.array_image, levels, **params)
            else:
                cs = self.ax.contour(self.x_axe_contours, self.y_axe_contours,
                                     self.array_contours, levels, **params)

            if len(self.clabels) > 0:
                dic_labels: dict = {}
                for (n, l) in zip(self.levels, self.clabels):
                    dic_labels[n] = l
                if len(self.clabels_mask) > 0:
                    self.ax.clabel(cs, self.levels[self.clabels_mask], fmt=dic_labels, **self.param_labels_contours)
                else:
                    self.ax.clabel(cs, self.levels, fmt=dic_labels, **self.param_labels_contours)
            else:
                self.ax.clabel(cs, **self.param_labels_contours)
            if self.title != "":
                self.ax.set_title(self.title)

    def projection_polygones(self) -> None:
        with mpl.rc_context(self.param_font):
            for i in range(len(self.index_polygons)):
                P = Path(self.index_polygons[i])
                poly = PathPatch(P, **self.param_polygons[i])
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
        with mpl.rc_context(self.param_font):
            self.param_ax["title"] = self.title
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
            if len(self.x_axe) == 0 or self.x_axe[0] > -np.inf:
                if len(self.x_axe) > 0:
                    self.ax.set_xlim([self.x_axe.min(), self.x_axe.max()])
                self.ax.set_xticks(self.x_axe)
            if len(self.labels_x_ticks) == 0 or self.labels_x_ticks[0] != "empty":
                self.ax.set_xticklabels(self.labels_x_ticks)
            if len(self.y_axe) == 0 or self.y_axe[0] > -np.inf:
                if len(self.y_axe) > 0:
                    self.ax.set_ylim([self.y_axe.min(), self.y_axe.max()])
                self.ax.set_yticks(self.y_axe)
            if len(self.labels_y_ticks) == 0 or self.labels_y_ticks[0] != "empty":
                self.ax.set_yticklabels(self.labels_y_ticks)
            if self.title != "":
                self.ax.set_title(self.title)

            if len(self.lines_x) > 0:
                self.projection_lines()
            if len(self.vals_histogramme) > 0:
                self.projection_histogrammes()
            if len(self.array_image) > 1:
                self.projection_image()
            if len(self.array_contours) > 1 or (self.tab_contours_is_image and len(self.array_image) > 1):
                self.projection_contours()
            if len(self.index_polygons) > 0:
                self.projection_polygones()
            if len(self.lines_t_x) > 0:
                self.projection_texts()
            if self.custum_colorbar_colors is not None:
                self.projection_colorbar()

            if (np.any(["label" in self.param_lines[i] for i in range(len(self.param_lines))])
                    or np.any(["label" in self.param_histogrammes[i] for i in range(len(self.param_histogrammes))])
                    or np.any(["label" in self.param_polygons[i] for i in range(len(self.param_polygons))])):
                plt.legend(**self.param_legende)
            if self.grid:
                plt.grid()

    def enregistrement_figure(self, **args) -> None:
        args_enrg = self.param_enrg_fig.copy()
        args_enrg.update(args)
        self.projection_figure()
        plt.savefig(self.directory + "/" +
                    self.filename + self.ext, **args_enrg)
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


def line(x, y=None, marker: str = "", **args) -> Graphique:
    """ Équivalent plt.plot;plt.show() """
    graph: Graphique = Graphique()
    graph.line(x, y, marker, **args)
    graph.affichage_figure()
    return graph


def incertitudes(x, y, err_y, marker: str = "", échelle: str = "",
                 **args) -> Graphique:
    """ Équivalent plt.errorbar(x,y,err_y,marker="",**args);plt.show() """
    graph: Graphique = Graphique()
    graph.errorbar(x, y, err_y, marker, échelle, **args)
    graph.affichage_figure()
    return graph


def incertitudes2(x, y, err_y, marker: str = "", échelle: str = "",
                  **args) -> Graphique:
    """ Équivalent affichage_incertitudes,
    mais avec un fond coloré à la place des barres d'erreur """
    graph: Graphique = Graphique()
    graph.errorplot(x, y, err_y, marker, échelle, **args)
    graph.affichage_figure()
    return graph


def polaire(R, Theta, **args) -> Graphique:
    """ Équivalent à plt.plot en projection polaire
    Attention, ordre opposé à celui de matplotlib :
        on indique le rayon puis l'angle et non l'inverse..."""
    graph: Graphique = Graphique()
    graph.polar(R, Theta, **args)
    graph.affichage_figure()
    return graph


def loglog(x, y, marker: str = "", **args) -> Graphique:
    """ Équivalent plt.loglog;plt.show()
    """
    graph: Graphique = Graphique()
    graph.loglog(x, y, marker, **args)
    graph.affichage_figure()
    return graph


def symloglog(x, y, marker: str = "", **args) -> Graphique:
    """ Équivalent plt.plot;plt.show()
    avec symlog en x symlog en y : affiche les valeurs positives et négatives """
    graph: Graphique = Graphique()
    graph.symloglog(x, y, marker, **args)
    graph.affichage_figure()
    return graph


def logx(x, y, marker: str = "", **args) -> Graphique:
    """ Équivalent plt.semilogx;plt.show() """
    graph: Graphique = Graphique()
    graph.logx(x, y, marker, **args)
    graph.affichage_figure()
    return graph


def symlogx(x, y, marker: str = "", **args) -> Graphique:
    """ Équivalent plt.plot;plt.show()
    avec symlog en x : affiche les valeurs positives et négatives """
    graph: Graphique = Graphique()
    graph.symlogx(x, y, marker, **args)
    graph.affichage_figure()
    return graph


def logy(x, y, marker: str = "", **args) -> Graphique:
    """ Équivalent plt.semilogy;plt.show() """
    graph: Graphique = Graphique()
    graph.logy(x, y, marker, **args)
    graph.affichage_figure()
    return graph


def symlogy(x, y, marker: str = "", **args) -> Graphique:
    """ Équivalent `plt.plot`, `plt.show()`
    avec symlog en x : affiche les valeurs positives et négatives """
    graph: Graphique = Graphique()
    graph.symlogy(x, y, marker, **args)
    graph.affichage_figure()
    return graph


def histogramme(valeurs, poids=None, normalisation: bool = True,
                statistic: str = 'sum',
                bins: int = 10, **args) -> Graphique:
    """ Équivalent plt.hist;plt.show() mais avec des bins centrés  """
    if poids is None:
        poids = []
    graph: Graphique = Graphique()
    graph.histogram(
        valeurs, poids, normalisation, statistic, bins, **args)
    graph.affichage_figure()
    return graph


def image(tableau, x_axe, y_axe, **args) -> Graphique:
    graph: Graphique = Graphique()
    graph.image(tableau, x_axe, y_axe, **args)
    graph.affichage_figure()
    return graph


def level_surface(x: np.ndarray | list, y: np.ndarray | list,
                  vals: np.ndarray | list, npix_x: int = 400, npix_y: int = 400,
                  logx: bool = False, logy: bool = False,
                  method: str = 'cubic', log_vals: bool = False, **args) -> Graphique:
    """Retourne une image represantant la courbe de niveau 2d associée aux 
    points définits par x, y, vals
    x: np.ndarray | list liste de taille n et dimension 1
    contenant les abscissa des points
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
