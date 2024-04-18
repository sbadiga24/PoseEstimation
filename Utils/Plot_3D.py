import sys
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QObject, pyqtSignal, QThread
import threading
import time
import pyqtgraph.opengl as gl


class SignalEmitter(QObject):
    """Emit signals with new data."""
    newData = pyqtSignal(np.ndarray, np.ndarray)


class DataGeneratorThread(threading.Thread):
    """Background thread for generating data."""

    def __init__(self, emitter, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emitter = emitter
        self.daemon = True  # Ensures thread exits when main program exits

    def run(self):
        while True:
            actual = np.random.rand(3) * 10
            predicted = np.random.rand(3) * 10
            self.emitter.newData.emit(actual, predicted)
            time.sleep(1 / 60)


class RealTimePlotter:
    """Main application for real-time plotting."""

    def __init__(self):
        self.app = QApplication(sys.argv)
        self.window = self.setup_ui()
        self.actualData = []
        self.predictedData = []
        self.emitter = SignalEmitter()
        self.emitter.newData.connect(self.update_plot)

        # Start data generation thread
        self.data_thread = DataGeneratorThread(self.emitter)
        self.data_thread.start()

    def setup_ui(self):
        """Set up the GUI."""
        w = gl.GLViewWidget()
        w.setWindowTitle('Real-time 3D Plotting of Actual vs. Predicted Points')
        w.setCameraPosition(distance=100)
        w.setBackgroundColor('grey')

        grid = gl.GLGridItem()
        grid.scale(1, 1, 1)
        w.addItem(grid)

        self.actualScatter = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=(1, 0, 0, 1), size=5)
        w.addItem(self.actualScatter)

        self.predictedScatter = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=(0, 1, 0, 1), size=5)
        w.addItem(self.predictedScatter)

        axis = gl.GLAxisItem()
        axis.setSize(1, 1, 1)
        w.addItem(axis)

        w.show()
        return w

    def update_plot(self, actual, predicted):
        """Update plot with new data."""
        self.actualData.append(actual)
        self.predictedData.append(predicted)
        self.actualScatter.setData(pos=np.array(self.actualData), color=(1, 0, 0, 1), size=5)
        self.predictedScatter.setData(pos=np.array(self.predictedData), color=(0, 1, 0, 1), size=5)

    def run(self):
        """Run the application."""
        sys.exit(self.app.exec_())


if __name__ == '__main__':
    plotter = RealTimePlotter()
    plotter.run()
