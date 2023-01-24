"""Create live plots for analysis on image sequences"""

# Nonstandard
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class AnimatedPlot:
    """Class to manage animated plots for live views"""

    def __init__(self, analysis=None, blit=False):
        """Parameters:

        - analysis: Analysis image series object (e.g. ContourTracking)
        - blit: if True, use blitting for fast rendering
        """
        self.analysis = analysis
        self.plot_init_done = False
        self.create_plot()
        self.animation = FuncAnimation(fig=self.fig,
                                       func=self.run_animation,
                                       frames=self.analysis.nums,
                                       cache_frame_data=False,
                                       repeat=False,
                                       blit=blit)

    def create_plot(self):
        """Can be overriden in subclass, but has to define at least self.fig"""
        self.fig, self.ax = plt.subplots()

    def first_plot(self, data):
        """What to do the first time data arrives on the plot.

        self.updated_artists must be defined here.
        """
        pass

    def update_plot(self, data):
        """What to do upon iterations of the plot after the first time."""
        pass

    def run_animation(self, num):
        """How to plot data during live views of analysis."""
        data = self.analysis.live_analysis(num)

        if not self.plot_init_done:
            self.first_plot(data)
            self.plot_init_done = True
        else:
            self.update_plot(data)

        return self.updated_artists
