# -*- coding: utf-8 -*-
"""
Module Name: calibrate_gcp.py
Description: Tk application to set camera GCP calibration

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
import csv
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
        self.root.title("Camera GCP Tool")
        self.canvas = tk.Canvas(root, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.menu_bar = Menu(root)
        root.config(menu=self.menu_bar)
        file_menu = Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(label="Open GCP Image", command=self.load_image)
        file_menu.add_command(label="Open Camera Calibration", command=self.load_camera_calibration)
        file_menu.add_command(label="Save GCPs", command=self.save_clicks)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.do_quit)
        options_menu = Menu(self.menu_bar, tearoff=0)
        options_menu.add_command(label="Reset Waypoints", command=self.reset_waypoints)
        self.menu_bar.add_cascade(label="File", menu=file_menu)
        self.menu_bar.add_cascade(label="Options", menu=options_menu)
        self.canvas.bind("<Button-1>", self.record_click)
        self.create_palette()
        self.pt: ProjectionTransformer = ProjectionTransformer()

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
            answer = askyesno(title='confirmation', message='Are you sure that you want to reset GCPs?', icon='question')
            if answer:
                self.canvas.config(width=self.photo.width(), height=self.photo.height())
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
                self.click_positions.clear()
                self.update_palette()
        else:
            messagebox.showinfo("Camera GCP Tool", "You need to load an image first", icon="info")

    def create_palette(self):
        self.palette = tk.Toplevel(self.root)
        self.palette.title("GCPs")
        self.palette.geometry("200x120")
        self.click_listbox = tk.Listbox(self.palette, height=10)
        self.click_listbox.pack(fill=tk.BOTH, expand=True)
        self.palette.attributes("-topmost", True)

    def update_palette(self):
        self.click_listbox.delete(0, tk.END)  # Clear the listbox
        for index, pos in enumerate(self.click_positions):
            self.click_listbox.insert(tk.END, f"GCP-{index} = {pos[0]}, {pos[1]}")

    def load_camera_calibration(self):
        self.source_filename = filedialog.askopenfilename()
        if self.source_filename:
            self.pt.load(self.source_filename)

    def load_image(self):
        self.source_filename = filedialog.askopenfilename()
        if self.source_filename:
            self.photo = ImageTk.PhotoImage(Image.open(self.source_filename))
            self.set_geometry(self.photo.width(), self.photo.height())
            self.canvas.config(width=self.photo.width(), height=self.photo.height())
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.click_positions.clear()
            self.update_palette()

    def record_click(self, event):
        if self.photo:
            self.click_positions.append((event.x, event.y))
            self.canvas.create_oval( (event.x - 1),  (event.y - 1), (event.x + 1), (event.y + 1), fill="#00ff00")
            self.canvas.create_oval( (event.x - 2),  (event.y - 2), (event.x + 2), (event.y + 2), fill="#ff0000")
            self.update_palette()

    def save_clicks(self):
        fieldnames = ["gcp", "x", "z"]
        if self.click_positions:
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
            if file_path:
                with open(file_path, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for index, xy in enumerate(self.click_positions):
                        xz = self.pt.get_xz(xy[0],xy[1])
                        writer.writerow({
                            "gcp": index,
                            "x": xz[0],
                            "z": xz[1]
                        })
                    messagebox.showinfo("Camera GCP Tool", "Saved", icon="info")
        else:
            messagebox.showinfo("Camera GCP Tool", "Could not save. You need GCPs.", icon="error")

if __name__ == "__main__":
    root = tk.Tk()
    app = CalibrateCamera(root)
    root.mainloop()
