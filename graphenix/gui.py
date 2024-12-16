import matplotlib
from matplotlib.backends.backend_qt import NavigationToolbar2QT

matplotlib.use('Qt5Agg')
import sys
from PyQt5 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from graphenix.objet import *


class PersoNavBar(NavigationToolbar2QT):
    def __init__(self, *args, **kwargs):
        super(PersoNavBar, self).__init__(*args, **kwargs)

    def zoom(self, *args):
        print("args zoom", args)
        super().zoom(*args)


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, graph: Graphique):
        self.axes = graph.axes
        print(self.axes)
        super().__init__(graph.fig)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, graph: Graphique, *args, **kwargs):
        super().__init__(*args, **kwargs)

        graph.plot()
        sc = MplCanvas(graph)
        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        # toolbar = NavigationToolbar(sc, self)
        toolbar = PersoNavBar(sc, self)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(sc)

        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.show()


def plot(graph: Graphique):
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(graph)
    app.exec_()


if __name__ == '__main__':
    args = sys.argv
    if len(args) > 0:
        gr = Graphique(args[1])
        plot(gr)