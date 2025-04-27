# -*- coding: utf-8 -*-
"""
Module Name: calibrate_camera.py
Description: Tk application to set camera waypoints

Copyright (C) 2025 J.Cincotta

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
import tkinter as tk
from tkinter import filedialog, Menu, messagebox, simpledialog
from tkinter.messagebox import askyesno
from PIL import Image, ImageTk
import numpy as np
import sys
import os

# hack to get bin folder to be able to access welfareobs
sys.path.insert(0, '../welfareobs')
from welfareobs.utils.projection_transformer import ProjectionTransformer
from welfareobs.utils.image_wrapper import ImageWrapper


class CalibrateCamera:
    def __init__(self, root):
        self.click_positions = []
        self.photo = None
        self.source_filename = None
        self.root = root
        self.set_geometry(800,600)
        self.root.title("Camera Calibration Tool")
        self.orientations = ["NW","NE","SW","SE"]
        self.canvas = tk.Canvas(root, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.menu_bar = Menu(root)
        root.config(menu=self.menu_bar)
        file_menu = Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.load_image)
        file_menu.add_command(label="Save Calibration", command=self.save_clicks)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.do_quit)
        options_menu = Menu(self.menu_bar, tearoff=0)
        options_menu.add_command(label="Reset Waypoints", command=self.reset_waypoints)
        options_menu.add_separator()
        options_menu.add_command(label="Preview View", command=self.warp_preview)
        self.menu_bar.add_cascade(label="File", menu=file_menu)
        self.menu_bar.add_cascade(label="Options", menu=options_menu)
        self.canvas.bind("<Button-1>", self.record_click)
        self.create_palette()

    def do_quit(self):
        if askyesno(title='confirmation', message='Are you sure?', icon='question'):
            self.root.quit()

    def set_geometry(self, w, h):
        ws = self.root.winfo_screenwidth()  # width of the screen
        hs = self.root.winfo_screenheight()  # height of the screen
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)
        self.root.geometry('%dx%d+%d+%d' % (w, h, x, y))

    def reset_waypoints(self):
        if self.photo:
            answer = askyesno(title='confirmation', message='Are you sure that you want to reset waypoints?', icon='question')
            if answer:
                self.canvas.config(width=self.photo.width(), height=self.photo.height())
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
                self.click_positions.clear()
                self.update_palette()
        else:
            messagebox.showinfo("Camera Calibration Tool", "You need to load an image first", icon="info")

    def create_palette(self):
        self.palette = tk.Toplevel(self.root)
        self.palette.title("Waypoints")
        self.palette.geometry("200x120")
        self.click_listbox = tk.Listbox(self.palette, height=4)
        self.click_listbox.pack(fill=tk.BOTH, expand=True)
        self.palette.attributes("-topmost", True)

    def update_palette(self):
        self.click_listbox.delete(0, tk.END)  # Clear the listbox
        for waypoint, pos in zip(self.orientations, self.click_positions[-4:]):  # Show only last 4 clicks
            self.click_listbox.insert(tk.END, f"{waypoint} = {pos[0]}, {pos[1]}")

    def load_image(self):
        self.source_filename = filedialog.askopenfilename()
        if self.source_filename:
            self.photo = ImageTk.PhotoImage(Image.open(self.source_filename))
            self.set_geometry(self.photo.width(), self.photo.height())
            self.canvas.config(width=self.photo.width(), height=self.photo.height())
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.click_positions.clear()
            self.update_palette()
            messagebox.showinfo("Image Calibration Tool", "You need to mark the four waypoints in a SQUARE by clicking the mouse. NW, NE, SW, SE.", icon="info")

    def record_click(self, event):
        if self.photo:
            if len(self.click_positions) < 4:
                self.click_positions.append((event.x, event.y))
                self.canvas.create_oval( (event.x - 1),  (event.y - 1), (event.x + 1), (event.y + 1), fill="#00ff00")
                self.canvas.create_oval( (event.x - 2),  (event.y - 2), (event.x + 2), (event.y + 2), fill="#ff0000")
                self.update_palette()

    def get_projection(self):
        pt: ProjectionTransformer = ProjectionTransformer()
        pt.calibrate(
            self.photo.width(),
            self.photo.height(),
            self.click_positions[-4],
            self.click_positions[-3],
            self.click_positions[-2],
            self.click_positions[-1]
        )
        return pt

    def warp_preview(self):
        if self.click_positions:
            pt: ProjectionTransformer = self.get_projection()
            pt.render_warp_map(ImageWrapper(self.source_filename, self.photo.width(), self.photo.height()))
        else:
            messagebox.showinfo("Image Calibration Tool", "Could not preview. You need to set all four way points, as well as the tile-count for NS and EW", icon="error")

    def save_clicks(self):
        if self.click_positions:
            pt: ProjectionTransformer = self.get_projection()
            file_path = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("PKL Files", "*.pkl")])
            if file_path:
                pt.save(file_path)
                messagebox.showinfo("Image Calibration Tool", "Saved", icon="info")
        else:
            messagebox.showinfo("Image Calibration Tool", "Could not save. You need to set all four way points, as well as the tile-count for NS and EW", icon="error")

if __name__ == "__main__":
    root = tk.Tk()
    app = CalibrateCamera(root)
    root.mainloop()
