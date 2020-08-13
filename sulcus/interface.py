import sys
sys.path.insert(1, '../functions')

import general
import vtk_functions

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pickle_functions import open_pickle, save_pickle

import os
import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox

from Otolith import Otolith
from enum import Enum


""" TODO:
    - disable menus when otolith is not set yet.
    CLEN!!
    # ADD PEAKS MAP?? (for coloring)
    FLAKE 8
"""


SULCUS_VAL = 255

class VM(Enum):
    count_nzp = 1
    vtk = 2


class Interface:
    def __init__(self, master):
        self.root = master
        self.root.title("Sulcus analysis interface")

        # Init empty vars
        self.otolith = Otolith()
        self.peaks_file_name = ""
        self.current_z = 0
        self.middle_z = None

        # TODO: seems redundant
        self.voxel_resolution = tk.DoubleVar(value=1)

        self.menubar = None
        self.open_window = None

        # Options
        self.option_interpolate_sulcus = tk.BooleanVar(value=True)
        self.option_top_edge = tk.BooleanVar(value=False)

        # PDA
        self.pda_wl = tk.IntVar(value="21")
        self.pda_increment = tk.IntVar(value="5")
        self.pda_k = tk.IntVar(value="2")
        self.pda_run_for_current_slice = tk.BooleanVar(value=False)
        self.show_progress = tk.BooleanVar(value=False)

        #
        self.run_loop_var = tk.BooleanVar(value=False)
        self.run_button = None

        # Top frame displaying otolith's name
        self.frame_title = tk.Frame(self.root, width=200, pady=15)
        self.frame_title.grid(row=0, column=0)
        self.otolith_label = tk.Label(self.frame_title)
        self.otolith_label.grid(row=0, column=0)

        # Frame containing the image
        self.fig = plt.Figure()
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor([0, 0, 0])
        self.img = None
        self.img_edge, = self.axes.plot([], [], ".",
                                        markersize=3, color="pink")

        self.img_peaks, = self.axes.plot([], [], "o", color="red")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        self.canvas.mpl_connect('button_press_event', self.img_click_event)

        # Navigation frame (bottom)
        self.frame_nav = tk.Frame(self.root, pady=15)
        self.frame_nav.grid(row=3, column=0, sticky="ew")

        tk.Button(self.frame_nav, text="-10",
                  command=lambda: self.img_inc(-10)).grid(row=0, column=1)
        tk.Button(self.frame_nav, text="-1",
                  command=lambda: self.img_inc(-1)).grid(row=0, column=2)

        self.slice_nr = tk.Label(self.frame_nav, padx=15)
        self.slice_nr.grid(row=0, column=3)

        tk.Button(self.frame_nav, text="+1",
                  command=lambda: self.img_inc(1)).grid(row=0, column=4)
        tk.Button(self.frame_nav, text="+10",
                  command=lambda: self.img_inc(10)).grid(row=0, column=5)

        # Centers buttons in frame_nav
        self.frame_nav.grid_columnconfigure(0, weight=1)
        self.frame_nav.grid_columnconfigure(6, weight=1)

        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        """ Setup interface """
        self.add_keybinds()
        self.add_menu()

        # """ DEBUG """

        oto = Otolith()
        oto.read_from_folder("/home/steven/scriptie/INPUTS/sphere_r_100")
        oto.name = "sphere"
        self.set_otolith(oto)
        self.img_load

        # otolith = Otolith()
        # slices = open_pickle("/home/steven/Documents/school/scriptie/pickle_files/sphere_pickle.pkl")
        # otolith.set_slices(slices)
        # otolith.name = "SPHERE"
        # self.set_otolith(otolith)
        # self.img_load()

        # otolith = Otolith()
        # # slices = np.zeros((100, 100, 100))
        # slices = open_pickle("/home/steven/Documents/school/scriptie/pickle_files/otoI48_pickle.pkl")
        # otolith.set_slices(slices)
        # otolith.name = "otoI48_DEBUG"
        # self.set_otolith(otolith)
        # self.img_load()

        # # BIG (OTOF73)
        # otolith = Otolith()
        # otolith.set_slices(open_pickle("/home/steven/Documents/school/scriptie/pickle_files/otoF73_pickle.pkl"))
        # otolith.name = "otoF73_DEBUG"


        # self.set_otolith(otolith)
        # self.img_load()

    def add_keybinds(self):
        self.root.bind('<w>', lambda x: self.img_inc(-1))
        self.root.bind('<e>', lambda x: self.img_inc(1))
        self.root.bind('<Left>', lambda x: self.img_inc(-1))
        self.root.bind('<Right>', lambda x: self.img_inc(1))
        self.root.bind('<s>', lambda x: self.img_inc(-10))
        self.root.bind('<d>', lambda x: self.img_inc(10))
        self.root.bind('<f>', lambda x: self.window_go_to())
        self.root.bind('<m>', lambda x: self.img_load_z(self.middle_z))
        self.root.bind('<Control-s>', lambda x: self.file_save())
        self.root.bind('<Control-Shift-KeyPress-S>',
                       lambda x: self.file_save_as())
        self.root.bind('<Delete>', lambda x: self.file_delete_peaks_slice())
        self.root.bind('<BackSpace>', lambda x: self.file_delete_peaks())
        self.root.bind('<q>', lambda x: self.exit())

    def add_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        self.root.option_add('*tearOff', False)

        # FILE menu
        menu_file = tk.Menu(menubar)
        menu_file.add_command(label="Open image stack",
                              command=self.file_open_folder)
        menu_file.add_separator()
        menu_file.add_command(label="Save peaks-map", accelerator="Ctrl+S",
                              command=self.file_save)
        menu_file.add_command(label="Save peaks-map as...",
                              accelerator="Ctrl+Shift+S",
                              command=self.file_save_as)
        menu_file.add_separator()
        menu_file_export_id = tk.Menu(menubar)
        menu_file.add_command(label="Load peaks-map",
                              command=self.file_open_peaks)
        menu_file.add_cascade(label="Export", menu=menu_file_export_id)
        menu_file_export_id.add_command(label="Sulcus image stack",
                                     command=lambda: self.file_export_id(False))
        menu_file_export_id.add_command(label="Otolith + sulcus image stack",
                                     command=lambda: self.file_export_id(True))
        menu_file_export_id.add_separator()
        menu_file_export_id.add_command(label="Sulcus polydata",
                                     command=lambda: self.file_export_pd(False))
        menu_file_export_id.add_command(label="Otolith polydata",
                                     command=lambda: self.file_export_pd(True))

        menu_file.add_separator()
        menu_file.add_command(label="Quit", accelerator="Q", command=self.exit)

        # VIEW menu
        menu_view = tk.Menu(menubar)
        menu_view.add_command(label="Go to slice...", accelerator="F",
                              command=self.window_go_to)
        menu_view.add_command(label="Go to middle slice", accelerator="M",
                              command=lambda: self.img_load_z(self.middle_z))
        menu_view.add_separator()
        menu_view.add_command(label="Go 10 slices back", accelerator="S",
                              command=lambda: self.img_inc(-10))
        menu_view.add_command(label="Go 1 slice back", accelerator="W",
                              command=lambda: self.img_inc(-1))
        menu_view.add_command(label="Go 1 slice forward", accelerator="E",
                              command=lambda: self.img_inc(1))
        menu_view.add_command(label="Go 10 slices forward", accelerator="D",
                              command=lambda: self.img_inc(10))
        menu_view.add_separator()
        menu_view.add_checkbutton(label="Show interpolated sulcus",
                                  command=self.img_load,
                                  variable=self.option_interpolate_sulcus)
        menu_view.add_checkbutton(label="Show top-edge",
                                  command=self.img_load,
                                  variable=self.option_top_edge)

        # RUN menu
        menu_run = tk.Menu(menubar)
        menu_run.add_command(label="PDA...", command=self.window_pda)

        # MEASURE menu
        menu_measure = tk.Menu(menubar)
        menu_measure_vol = tk.Menu(menubar)
        menu_measure.add_cascade(label="Volume...", menu=menu_measure_vol)
        menu_measure_vol.add_command(label="Volume of otolith", command=lambda: self.window_measure("Volume of otolith", lambda: self.measure_vol_func(0)))
        menu_measure_vol.add_command(label="Volume of (current) sulcus", command=lambda: self.window_measure("Volume of sulcus", lambda: self.measure_vol_func(1)))
        menu_measure_surface = tk.Menu(menubar)
        menu_measure_surface.add_command(label="Surface area of otolith",
                                         command=lambda: self.window_measure("Surface area of otolith",
                                                 lambda: self.measure_surface_func(0)))
        menu_measure_surface.add_command(label="Proximal surface area of otolith",
                                         command=lambda: self.window_measure("Proximal surface area of otolith",
                                                 lambda: self.measure_surface_func(0, 1)))
        menu_measure_surface.add_separator()
        menu_measure_surface.add_command(label="Surface area of sulcus",
                                         command=lambda: self.window_measure("Surface area of sulcus",
                                                 lambda: self.measure_surface_func(1)))
        menu_measure_surface.add_command(label="Proximal surface area of sulcus",
                                         command=lambda: self.window_measure("Proximal surface area of sulcus",
                                                 lambda: self.measure_surface_func(1, -1)))
        menu_measure.add_cascade(label="Surface area...", menu=menu_measure_surface)

        # Add menu's to menubar
        menubar.add_cascade(label="File", menu=menu_file)
        menubar.add_cascade(label="View", menu=menu_view)
        menubar.add_cascade(label="Run", menu=menu_run)
        menubar.add_cascade(label="Measure", menu=menu_measure)

        self.menubar = menubar

    """ Setters & Getters """

    # Get z of neighbor in direction within max_distance
    def get_neighbor_peaks(self, direction, max_distance=10):
        d = self.otolith.get_neighbor_d(self.current_z, direction,
                                        max_distance)

        if d:
            return self.otolith.peaks.get(self.current_z + d * direction)
        return None

    # Set global otolith to otolith parameter
    def set_otolith(self, otolith, padding=30):
        self.otolith = otolith

        self.middle_z = int(self.otolith.slices.shape[2] / 2)
        self.current_z = self.middle_z

        width, height = self.otolith.slices[:, :, 0].shape

        self.axes.set_xlim(-padding, height + padding)
        self.axes.set_ylim(-padding, width + padding)
        self.axes.invert_yaxis()
        self.img = self.axes.imshow(np.zeros((width, height), dtype=np.uint8),
                                    vmin=0, vmax=SULCUS_VAL, cmap="gray")

        self.otolith_label.config(text="%s" % self.otolith.name)

        print("Set otolith to \"%s\"" % self.otolith.name)

    def set_peaks_from_filename(self, filename):
        obj = open_pickle(filename)
        self.set_peaks(obj)
        self.peaks_file_name = filename
        print("Loaded peaks from %s" % filename)

    def set_peaks(self, peaks):
        self.otolith.set_peaks(peaks)
        print("Set peaks")

    def peak_add(self, z, peak):
        if self.otolith.peaks.get(z):
            self.otolith.peaks[z].append(peak)
            peaks_xs, peaks_ys = zip(*self.otolith.peaks[z])

            sorted_indices = np.argsort(peaks_xs)

            peaks_xs = np.array(peaks_xs)
            peaks_ys = np.array(peaks_ys)

            self.otolith.peaks[z] = list(zip(peaks_xs[sorted_indices],
                                             peaks_ys[sorted_indices]))
        else:
            self.otolith.peaks[z] = [peak]

    def peak_remove(self, z, peak):
        index = self.otolith.peaks[z].index(peak)
        self.otolith.peaks[z].pop(index)

        if len(self.otolith.peaks[z]) == 0:
            self.delete_peaks_slice(z)

    """ USER functions """
    # PDA funcs
    def window_pda(self):
        self.open_window = tk.Toplevel(self.root)
        self.open_window.title("PDA parameters")
        self.open_window.resizable(0, 0)

        tk.Label(self.open_window, text="Window length (m):") \
            .grid(row=0, column=0, padx=15, pady=(15, 0))
        tk.Entry(self.open_window, textvariable=self.pda_wl, width=5) \
            .grid(row=0, column=1, padx=15, pady=(15, 0))

        tk.Label(self.open_window, text="Increment (l):") \
            .grid(row=3, column=0, padx=15, pady=(15, 0))
        tk.Scale(self.open_window, variable=self.pda_increment, from_=1, to=10,
                 orient="horizontal") \
            .grid(row=3, padx=15, pady=(15, 0), column=1)

        tk.Label(self.open_window, text="k:") \
            .grid(row=4, column=0, padx=15, pady=(15, 0))
        tk.Scale(self.open_window, variable=self.pda_k, from_=1, to=10,
                orient="horizontal") \
            .grid(row=4, padx=15, pady=(15, 0), column=1)

        tk.Checkbutton(self.open_window, text="Run for current slice only",
                       variable=self.pda_run_for_current_slice) \
            .grid(row=5, column=0, columnspan=2, padx=15, pady=(15, 0))
        tk.Checkbutton(self.open_window, text="Show progress",
                       variable=self.show_progress) \
            .grid(row=6, column=0, columnspan=2, padx=15, pady=(15, 0))

        self.run_button = tk.Button(self.open_window, text="Run PDA", command=self.pda_switch)
        self.run_button.grid(row=7, columnspan=2, padx=15, pady=15)

        self.open_window.bind('<Return>', lambda x: self.pda_switch())
        self.open_window.bind('<Escape>', lambda x: self.open_window.destroy())

        self.open_window.transient(self.root)
        self.open_window.grab_set()

        self.root.wait_window(self.open_window)

    """ FILE functions """
    def file_open_folder(self):
        folder = tk.filedialog.askdirectory(title="Open otolith image stack (folder)")

        if folder:
            otolith = Otolith()
            otolith.read_from_folder(folder)
            self.set_otolith(otolith)
            self.img_load()
        return

    def file_save(self):
        if not self.peaks_file_name:
            self.file_save_as()
        else:
            self.otolith.save_peaks(self.peaks_file_name)
            print("Saved peaks to %s" % self.peaks_file_name)

    def file_save_as(self):
        if self.peaks_file_name:
            initialfile = self.peaks_file_name
        else:
            initialfile = self.otolith.name + "_PEAKSMAP.pkl"

        filename = tk.filedialog.asksaveasfilename(title="Save peaks-map as...",
                                                   initialfile=initialfile,
                                                   filetypes=[("pkl", "*.pkl")])

        if filename:
            self.peaks_file_name = filename
            self.file_save()

        return

    def file_open_peaks(self):
        filename = tk.filedialog.askopenfilename(title="Load peaks-map",
                                                 filetypes=[("pkl", "*.pkl")])

        if filename and os.path.isfile(filename):
            self.set_peaks_from_filename(filename)

        self.img_load()

    def file_export_id(self, include_otolith=False):
        folder = tk.filedialog.askdirectory(title="Select export location")

        if folder:
            sulcus = self.otolith.get_sulcus() * 255

            if include_otolith:
                sulcus = self.otolith.slices + sulcus

            general.arr_to_imgseq(sulcus, "%s/%s_sulcus" % (folder, self.otolith.name), verbose=True)

    def file_export_pd(self, is_otolith=False):
        if is_otolith:
            initialfile = self.otolith.name + "_OTOLITH"
        else:
            initialfile = self.otolith.name + "_SULCUS"

        filename = tk.filedialog.asksaveasfilename(title="Save polydata as...",
                                                   initialfile=initialfile,
                                                   filetypes=[("ply", "*.ply")])

        if filename:
            if not is_otolith:
                polydata = self.otolith.get_isosurface_sulcus()
                vtk_functions.write_ply(polydata, filename)
            else:
                polydata = self.otolith.get_isosurface_otolith()
                vtk_functions.write_ply(polydata, filename)

            print("Got isosurface")

    def file_delete_peaks(self):
        self.otolith.peaks = {}

        print("Reset peaks for %s" % self.otolith.name)

        self.img_load()

    def file_delete_peaks_slice(self):
        if self.otolith.peaks.get(self.current_z):
            self.delete_peaks_slice(self.current_z)
            self.file_save()
            self.img_load()

    def delete_peaks_slice(self, z):
        del self.otolith.peaks[z]

    def exit(self):
        self.root.destroy()

    def get_current_peaks(self):
        if self.otolith.peaks.get(self.current_z):
            return self.otolith.peaks[self.current_z]
        return None

    """ IMG functions (canvas) """

    # Load image displayed on canvas frame
    def img_load(self, load_sulcus=True):
        self.slice_nr.config(text="%s / %s" % (self.current_z, self.otolith.slices.shape[2] - 1))

        self.img.set_array(self.otolith.slices[:, :, self.current_z])

        self.img_add_top_edge(self.option_top_edge.get())

        current_peaks = self.get_current_peaks()

        if current_peaks:
            peaks_xs, peaks_ys = zip(*current_peaks)
            self.img_peaks.set_data(peaks_xs, peaks_ys)

            self.otolith.peaks[self.current_z] = current_peaks

            if len(current_peaks) >= 2:
                self.img_add_sulcus()
        else:
            if self.option_interpolate_sulcus.get() and load_sulcus:
                self.img_add_sulcus()
            self.img_peaks.set_data([], [])

        self.canvas.draw()

    # Load image with index z
    def img_load_z(self, z):
        if z > self.otolith.slices.shape[2] - 1:
            self.current_z = self.otolith.slices.shape[2] - 1
        elif z < 0:
            self.current_z = 0
        else:
            self.current_z = z

        self.img_load()

    def img_add_top_edge(self, show=True):
        if show:
            img = self.otolith.slices[:, :, self.current_z]
            edge_xs, edge_ys = general.get_top_edge(img)

            self.img_edge.set_data(edge_xs, edge_ys)
        else:
            self.img_edge.set_data([], [])

    # Increment image with icn
    def img_inc(self, inc):
        self.img_load_z(self.current_z + inc)

    # Add sulcus area to current slice image
    def img_add_sulcus(self):
        sulcus_2d = self.otolith.get_sulcus_2d(self.current_z)

        if np.any(sulcus_2d):
            oto_sulcus_2d = self.otolith.slices[:, :, self.current_z] + (sulcus_2d * 255)
            self.img.set_array(oto_sulcus_2d)

    # Remove sulcus from slice img
    def img_remove_sulcus(self):
        self.img.set_array(self.otolith.slices[:, :, self.current_z])

    # Add peaks to Otolith on click
    def img_click_event(self, event, radius=15):
        if not (event.xdata or event.ydata):
            return

        img = self.otolith.slices[:, :, self.current_z]
        edge_xs, edge_ys = general.get_top_edge(img)

        x_val = event.xdata

        if x_val < min(edge_xs):
            x_val = min(edge_xs)
        elif x_val > max(edge_xs):
            x_val = max(edge_xs)

        x = int(round(x_val))
        y = general.get_y_val(x, edge_xs, edge_ys)

        change = False

        if self.get_current_peaks():
            # If current clicked value is within radius of existing peaks:
            # remove peak
            not_found = True

            for p in self.get_current_peaks():
                if abs(p[0] - x) < radius:
                    self.peak_remove(self.current_z, p)
                    change = True
                    not_found = False
                    break

            # If peaks were not in range, add peak to dictionary
            if not_found:
                self.peak_add(self.current_z, (x, y))
                change = True

        # If current slice has no points
        else:
            self.peak_add(self.current_z, (x, y))
            change = True

        if change:
            self.img_load(False)
            self.file_save()

    """ WINDOWS """
    def window_info(self, title, body, copy=None):
        top = tk.Toplevel(self.root)
        top.title(title)
        top.resizable(0, 0)
        field_body = tk.Text(top, wrap=tk.WORD, width=40, height=5)
        field_body.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
        field_body.insert(1.0, body)

        field_body.configure(bg=self.root.cget('bg'), relief="flat")
        field_body.configure(state="disabled")

        tk.Button(top, text="Copy to clipboard", command=lambda: self.copy_to_clipboard(copy)) \
            .grid(row=1, column=0, padx=15)

        tk.Button(top, text="Close", command=top.destroy) \
            .grid(row=2, column=0, padx=15, pady=15)

        top.bind('<Return>', lambda x: top.destroy())
        top.bind('<Escape>', lambda x: top.destroy())

        top.transient(self.root)
        top.grab_set()

    def copy_to_clipboard(self, body):
        self.root.clipboard_clear()
        self.root.clipboard_append(body)
        self.root.update()

    def window_go_to(self):
        self.open_window = tk.Toplevel(self.root)
        self.open_window.title("Go to slice...")
        self.open_window.resizable(0, 0)

        z_field = tk.Entry(self.open_window)
        z_field.grid(row=0, column=0, padx=15, pady=15)
        z_field.focus()

        self.open_window.bind('<Return>', lambda x: self.go_to_func(z_field.get()))
        self.open_window.bind('<Escape>', lambda x: self.open_window.destroy())

        self.open_window.transient(self.root)
        self.open_window.grab_set()
        self.root.wait_window(self.open_window)

    def window_measure(self, title, cmd):
        self.open_window = tk.Toplevel(self.root)
        self.open_window.title(title)
        self.open_window.resizable(0, 0)

        tk.Label(self.open_window, text="Voxel resolution (μm):") \
          .grid(row=0, padx=15, pady=(15, 0), column=0, columnspan=2)

        tk.Entry(self.open_window, textvariable=self.voxel_resolution) \
           .grid(row=1, column=0, padx=15, pady=(15, 0), sticky="ew")

        tk.Checkbutton(self.open_window, text="Show progress",
                       variable=self.show_progress) \
            .grid(row=4, column=0, columnspan=2, padx=15, pady=(15, 0))

        self.run_button = tk.Button(self.open_window, text="Start",
                                    command=lambda: self.measure_switch(cmd))
        self.run_button.grid(row=5, column=0, columnspan=2, padx=15, pady=15)

        self.open_window.bind('<Escape>', lambda x: self.open_window.destroy())

        self.open_window.transient(self.root)
        self.open_window.grab_set()

        self.root.wait_window(self.open_window)

    # USER CALLABLE FUNCS + helpers
    def pda_switch(self):
        if not self.run_loop_var.get():
            self.run_button.config(text="Stop PDA")
            self.run_loop_var.set(True)
            self.pda_func()
            self.run_loop_var.set(False)
        else:
            self.run_button.config(text="Run PDA")
            self.run_loop_var.set(False)

    def measure_switch(self, cmd):
        if not self.run_loop_var.get():
            self.run_button.config(text="Stop")
            self.run_loop_var.set(True)
            cmd()
            self.run_loop_var.set(False)
        else:
            self.run_button.config(text="Start")
            self.run_loop_var.set(False)

    def set_pda_params(self):
        self.otolith.wl = self.pda_wl.get()

    def pda_func(self):
        self.set_pda_params()
        self.file_save()

        wl = self.otolith.wl

        print("Running PDA in slice %s..." % self.current_z)
        print("WL: %s" % wl)

        if self.otolith.peaks.get(self.current_z):
            self.otolith.peaks[self.current_z] = []

        if self.pda_run_for_current_slice.get():
            peaks_detected = self.otolith.detect_peaks_2d(self.current_z)

            for p in peaks_detected:
                self.peak_add(self.current_z, p)

            self.img_load(self.current_z)
        else:
            peaks_generator = self.otolith.detect_peaks(inc=self.pda_increment.get(), max_nps=self.pda_k.get(), verbose=True)

            for z, peaks in peaks_generator:
                if not self.run_loop_var.get():
                    break

                self.current_z = z

                for p in peaks:
                    self.peak_add(z, p)

                if self.show_progress.get():
                    self.img_load()

                self.root.update()

        self.file_save()

        if not self.show_progress.get():
            self.img_load()

    def measure_vol_func(self, r, t=VM.count_nzp):
        volume_generator = None
        if r == 0:
            print("Calculating OTOLITH volume with %s" % self.voxel_resolution.get())
            volume_generator = self.otolith.get_nzp_otolith()
        elif r == 1:
            print("Calculating SULCUS volume with %s" % self.voxel_resolution.get())
            volume_generator = self.otolith.get_nzp_sulcus()

        total_nzp = 0

        if t == VM.vtk:
            total_nzp = self.otolith.get_volume_otolith_vtk()
        else:
            for z, vol in volume_generator:
                if not self.run_loop_var.get():
                    return

                if self.show_progress.get():
                    self.current_z = z

                    if r == 0:
                        self.img_load(load_sulcus=False)
                    else:
                        self.img_load()

                print("Slice %s: %s non-zero pixels" % (z, vol))

                total_nzp += vol
                self.root.update()

            self.run_loop_var

        vol = total_nzp * (self.voxel_resolution.get()**3)

        body = "Total non-zero pixels: %s\n" % total_nzp
        body += "Voxel resolution: %s\n" % self.voxel_resolution.get()
        body += "Volume: %s μm^3" % vol

        self.open_window.destroy()
        self.window_info("Measurements", body, copy=vol)

    def measure_surface_func(self, part, pos=0):
        s = 0

        if part == 0:
            if pos == 0:
                s = self.otolith.get_surface_area_otolith()
            elif pos == 1:
                s = self.otolith.get_proximal_surface_otolith()
        elif part == 1:
            if pos == 0:
                s = self.otolith.get_surface_area_sulcus()
            elif pos == -1:
                s = self.otolith.get_proximal_surface_sulcus()

        part_string = ""
        if part == 0:
            part_string = "otolith"
        elif part == 1:
            part_string = "sulucs"

        side_string = ""
        if pos == 1:
            side_string = "Proximal"
        elif pos == 0:
            side_string = ""
        elif pos == -1:
            side_string = "Distal"

        surface = s * (self.voxel_resolution.get()**2)

        body = "Surface area (voxels): %s\n" % s
        body += "Voxel resolution: %s\n" % self.voxel_resolution.get()
        body += "%s surface area of %s: %s μm^2" % (side_string, part_string, surface)

        self.open_window.destroy()
        self.window_info("Measurements", body, copy=surface)

    def go_to_func(self, z):
        try:
            z = int(z)
        except ValueError:
            print("Input is not an integer!")
            return

        self.img_load_z(z)
        self.open_window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = Interface(root)
    root.mainloop()
