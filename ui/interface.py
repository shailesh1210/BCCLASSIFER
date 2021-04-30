import wx
import os
import threading as td
import time

from core.train import train_model
from core.predict import run_predict_ui
from core.classifier import Classifier


class Interface(wx.Frame):
    HEIGHT = 900
    WIDTH = 1920

    def __init__(self):
        super().__init__(parent=None,
                         title="BCCLASSIFIER",
                         size=(Interface.WIDTH, Interface.HEIGHT),
                         style=wx.MINIMIZE_BOX | wx.SYSTEM_MENU | wx.CAPTION | wx.CLOSE_BOX)

        self.hyper_params = {'kernel_size': 5, 'kernel_stride': 1,
                             'pool_size': 3, 'pool_stride': 1, 'epochs': 100,
                             'optimizer': 'Adam', 'lr': 0.001, 'batch_size': 32}

        self.weights_path = list()
        self.history_path = list()
        self.image_paths = list()

    def initialize(self):
        self.window = wx.Panel(self)

        self.parent_box = wx.BoxSizer(wx.HORIZONTAL)

        self.left_box = wx.BoxSizer(wx.VERTICAL)
        self.right_box = wx.BoxSizer(wx.VERTICAL)

        self.__create_input_widgets()
        self.__create_output_widgets()

        self.parent_box.Add(self.left_box, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        self.parent_box.Add(self.right_box, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)

        self.window.SetSizer(self.parent_box)

        self.Center()
        self.Show()

    def __create_input_widgets(self):
        self.__create_choice_box()

        self.__create_training_menu()
        self.__create_predict_menu()

        self.__create_buttons()

    def __create_output_widgets(self):
        display_vbox = wx.BoxSizer(wx.VERTICAL)

        display_label = wx.StaticText(self.window, label="Output Console")
        self.display_text_box = wx.TextCtrl(self.window, style=wx.TE_MULTILINE | wx.TE_READONLY)

        self.image_panel = wx.Panel(self.window, style=wx.SUNKEN_BORDER)
        self.image_panel.SetBackgroundColour(colour=(255, 255, 255))

        display_vbox.Add(display_label, flag=wx.ALL | wx.ALIGN_CENTER, border=5)
        display_vbox.Add(self.display_text_box, proportion=1, flag=wx.EXPAND)
        display_vbox.AddSpacer(size=10)
        display_vbox.Add(self.image_panel, proportion=1, flag=wx.EXPAND)

        self.right_box.Add(display_vbox, proportion=1, flag=wx.EXPAND)

    def __create_choice_box(self):
        choice_vbox = wx.BoxSizer(wx.VERTICAL)

        model_label = wx.StaticText(self.window, label="Select from")

        self.models = ["Train", "Predict"]
        self.choice_models = wx.Choice(self.window, choices=self.models,
                                       name=self.models[0],
                                       size=(200, 30))
        self.choice_models.Bind(wx.EVT_CHOICE, self.__model_selection)

        choice_vbox.Add(model_label, flag=wx.ALL | wx.ALIGN_CENTER, border=10)
        choice_vbox.Add(self.choice_models, flag=wx.LEFT | wx.RIGHT | wx.EXPAND, border=10)

        self.left_box.Add(choice_vbox, flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=10)

    def __create_buttons(self):
        button_vbox = wx.BoxSizer(wx.VERTICAL)

        self.training_button = wx.Button(self.window, label="Train", size=(200, 40))
        self.Bind(wx.EVT_BUTTON, self.__thread_start, self.training_button)
        self.training_button.Disable()

        self.predict_button = wx.Button(self.window, label="Predict", size=(200, 40))
        self.Bind(wx.EVT_BUTTON, self.__run_prediction, self.predict_button)
        self.predict_button.Disable()

        self.clear_btn = wx.Button(self.window, label="Clear", size=(200, 40))
        self.clear_btn.Bind(wx.EVT_BUTTON, self.__clear)

        button_vbox.Add(self.training_button, flag=wx.RIGHT | wx.LEFT | wx.BOTTOM | wx.EXPAND, border=10)
        button_vbox.Add(self.predict_button, flag=wx.RIGHT | wx.LEFT | wx.BOTTOM | wx.EXPAND, border=10)
        button_vbox.Add(self.clear_btn, flag=wx.RIGHT | wx.LEFT | wx.EXPAND, border=10)

        self.left_box.Add(button_vbox, flag=wx.EXPAND | wx.ALL, border=10)

    def __create_training_menu(self):
        sizer = wx.GridSizer(9, 2, 10, 10)

        self.kernel_size_text = wx.StaticText(self.window, label="Kernel Size", style=wx.ALIGN_CENTER)
        self.kernel_size_value = wx.TextCtrl(self.window, value=str(self.hyper_params['kernel_size']), style=wx.ALIGN_RIGHT)
        self.kernel_size_value.Bind(wx.EVT_TEXT, self.__set_kernel_size)

        self.kernel_stride_text = wx.StaticText(self.window, label="Kernel Stride", style=wx.ALIGN_CENTER)
        self.kernel_stride_value = wx.TextCtrl(self.window, value=str(self.hyper_params['kernel_stride']), style=wx.ALIGN_RIGHT)
        self.kernel_stride_value.Bind(wx.EVT_TEXT, self.__set_kernel_stride)

        self.pool_size_text = wx.StaticText(self.window, label="Pooling Size", style=wx.ALIGN_CENTER)
        self.pool_size_value = wx.TextCtrl(self.window, value=str(self.hyper_params['pool_size']), style=wx.ALIGN_RIGHT)
        self.pool_size_value.Bind(wx.EVT_TEXT, self.__set_pool_size)

        self.pool_stride_text = wx.StaticText(self.window, label="Pooling Stride", style=wx.ALIGN_CENTER)
        self.pool_stride_value = wx.TextCtrl(self.window, value=str(self.hyper_params['pool_stride']), style=wx.ALIGN_RIGHT)
        self.pool_stride_value.Bind(wx.EVT_TEXT, self.__set_pool_stride)

        self.optimizer_text = wx.StaticText(self.window, label="Optimizer", style=wx.ALIGN_CENTER)
        self.optimizer_value = wx.TextCtrl(self.window, value=str(self.hyper_params['optimizer']), style=wx.ALIGN_RIGHT)
        self.optimizer_value.Bind(wx.EVT_TEXT, self.__set_optimizer)

        self.learning_rate_text = wx.StaticText(self.window, label="Learning Rate (LR)", style=wx.ALIGN_CENTER)
        self.learning_rate_value = wx.TextCtrl(self.window, value=str(self.hyper_params['lr']), style=wx.ALIGN_RIGHT)
        self.learning_rate_value.Bind(wx.EVT_TEXT, self.__set_learning_rate)

        self.batch_size_text = wx.StaticText(self.window, label="Batch Size", style=wx.ALIGN_CENTER)
        self.batch_size_value = wx.TextCtrl(self.window, value=str(self.hyper_params['batch_size']), style=wx.ALIGN_RIGHT)
        self.batch_size_value.Bind(wx.EVT_TEXT, self.__set_batch_size)

        self.epochs_text = wx.StaticText(self.window, label="Epochs", style=wx.ALIGN_CENTER)
        self.epochs_value = wx.TextCtrl(self.window, value=str(self.hyper_params['epochs']), style=wx.ALIGN_RIGHT)
        self.epochs_value.Bind(wx.EVT_TEXT, self.__set_epochs)

        sizer.AddMany([self.kernel_size_text, (self.kernel_size_value, 1, wx.EXPAND),
                       self.kernel_stride_text, (self.kernel_stride_value, 1, wx.EXPAND),
                       self.pool_size_text, (self.pool_size_value, 1, wx.EXPAND),
                       self.pool_stride_text, (self.pool_stride_value, 1, wx.EXPAND),
                       self.optimizer_text, (self.optimizer_value, 1, wx.EXPAND),
                       self.learning_rate_text, (self.learning_rate_value, 1, wx.EXPAND),
                       self.batch_size_text, (self.batch_size_value, 1, wx.EXPAND),
                       self.epochs_text, (self.epochs_value, 1, wx.EXPAND)])

        self.left_box.Add(sizer, flag=wx.EXPAND | wx.ALL, border=15)

        self.__hide_training_menu()

    def __create_predict_menu(self):
        sizer = wx.GridSizer(3, 2, 10, 10)

        self.load_weights_check_box = wx.CheckBox(self.window, label="Load Model")
        self.load_weights_check_box.Bind(wx.EVT_CHECKBOX, self.__enable_weights_import)

        self.load_history_check_box = wx.CheckBox(self.window, label="Load model history")
        self.load_history_check_box.Bind(wx.EVT_CHECKBOX, self.__enable_history_import)

        self.load_images_check_box = wx.CheckBox(self.window, label="Import Histopathological image(s)")
        self.load_images_check_box.Bind(wx.EVT_CHECKBOX, self.__enable_image_import)

        self.import_weights_btn = wx.Button(self.window, label="Import Model", size=(400, 40))
        self.import_weights_btn.Disable()
        self.import_weights_btn.Bind(wx.EVT_BUTTON, self.__import_weights)

        self.import_history_btn = wx.Button(self.window, label="Import Model History", size=(400, 40))
        self.import_history_btn.Disable()
        self.import_history_btn.Bind(wx.EVT_BUTTON, self.__import_history)

        self.import_image_btn = wx.Button(self.window, label="Import Images", size=(400, 40))
        self.import_image_btn.Disable()
        self.import_image_btn.Bind(wx.EVT_BUTTON, self.__import_images)

        sizer.AddMany([self.load_weights_check_box, (self.import_weights_btn, 1, wx.EXPAND),
                       self.load_history_check_box, (self.import_history_btn, 1, wx.EXPAND),
                       self.load_images_check_box, (self.import_image_btn, 1, wx.EXPAND)])

        self.left_box.Add(sizer, flag=wx.EXPAND | wx.ALL, border=15)

        self.__hide_predict_menu()

    def __model_selection(self, event):
        self.model_name = self.models[self.choice_models.GetCurrentSelection()]
        if self.model_name == "Train":
            self.__hide_predict_menu()
            self.__display_training_menu()

            self.predict_button.Disable()
            self.training_button.Enable()

        elif self.model_name == 'Predict':
            self.__hide_training_menu()
            self.__display_predict_menu()

            self.training_button.Disable()
            self.predict_button.Enable()

        self.display_text_box.AppendText(self.model_name + " is selected!\n")

        # self.choice_models.Disable()
        self.window.Fit()
        self.window.SetSize((Interface.WIDTH-10, Interface.HEIGHT-10))

    def __display_training_menu(self):
        self.kernel_size_text.Show()
        self.kernel_size_value.Show()

        self.kernel_stride_text.Show()
        self.kernel_stride_value.Show()

        self.pool_stride_text.Show()
        self.pool_stride_value.Show()

        self.pool_size_text.Show()
        self.pool_size_value.Show()

        self.optimizer_text.Show()
        self.optimizer_value.Show()

        self.learning_rate_text.Show()
        self.learning_rate_value.Show()

        self.batch_size_text.Show()
        self.batch_size_value.Show()

        self.epochs_text.Show()
        self.epochs_value.Show()

    def __display_predict_menu(self):
        self.load_images_check_box.Show()
        self.import_image_btn.Show()

        self.load_history_check_box.Show()
        self.import_history_btn.Show()

        self.load_weights_check_box.Show()
        self.import_weights_btn.Show()

        self.predict_button.Enable()

    def __hide_training_menu(self):
        self.kernel_size_text.Hide()
        self.kernel_size_value.Hide()

        self.kernel_stride_text.Hide()
        self.kernel_stride_value.Hide()

        self.pool_stride_text.Hide()
        self.pool_stride_value.Hide()

        self.pool_size_text.Hide()
        self.pool_size_value.Hide()

        self.optimizer_text.Hide()
        self.optimizer_value.Hide()

        self.learning_rate_text.Hide()
        self.learning_rate_value.Hide()

        self.batch_size_text.Hide()
        self.batch_size_value.Hide()

        self.epochs_text.Hide()
        self.epochs_value.Hide()

    def __hide_predict_menu(self):
        self.load_images_check_box.Hide()
        self.import_image_btn.Hide()

        self.load_history_check_box.Hide()
        self.import_history_btn.Hide()

        self.load_weights_check_box.Hide()
        self.import_weights_btn.Hide()

    def __run_training(self, event):
        if self.model_name == "Train":
            train_model(hyper_params=self.hyper_params)
            time.sleep(3)

    def __run_prediction(self, event):
        if self.model_name == "Predict":
            if len(self.weights_path) == 0 or len(self.image_paths) == 0 or len(self.history_path) == 0:
                self.display_text_box.AppendText('Please select model or model history or images!\n')

            preds = run_predict_ui(model_path=self.weights_path[0], history_path=self.history_path[0],
                                   image_paths=self.image_paths)

            self.__display_images(image_paths=self.image_paths, tags=preds)

    def __set_kernel_size(self, event):
        self.hyper_params['kernel_size'] = int(self.kernel_size_value.GetValue())

    def __set_kernel_stride(self, event):
        self.hyper_params['kernel_stride'] = int(self.kernel_stride_value.GetValue())

    def __set_pool_size(self, event):
        self.hyper_params['pool_size'] = int(self.pool_size_value.GetValue())

    def __set_pool_stride(self, event):
        self.hyper_params['pool_stride'] = int(self.pool_stride_value.GetValue())

    def __set_optimizer(self, event):
        self.hyper_params['optimizer'] = self.optimizer_value.GetValue()

    def __set_learning_rate(self, event):
        self.hyper_params['lr'] = float(self.learning_rate_value.GetValue())

    def __set_batch_size(self, event):
        self.hyper_params['batch_size'] = int(self.batch_size_value.GetValue())

    def __set_epochs(self, event):
        self.hyper_params['epochs'] = int(self.epochs_value.GetValue())

    def __enable_weights_import(self, event):
        if self.load_weights_check_box.IsChecked():
            self.import_weights_btn.Enable()
        else:
            self.import_weights_btn.Disable()

    def __enable_history_import(self, event):
        if self.load_history_check_box.IsChecked():
            self.import_history_btn.Enable()
        else:
            self.import_history_btn.Disable()

    def __enable_image_import(self, event):
        if self.load_images_check_box.IsChecked():
            self.import_image_btn.Enable()
        else:
            self.import_image_btn.Disable()

    def __import_weights(self, event):
        self.__clear_weight_paths()
        self.weights_path = self.__import_files()

    def __import_history(self, event):
        self.__clear_history_paths()
        self.history_path = self.__import_files()

    def __import_images(self, event):
        self.image_paths = self.__import_files()
        self.__display_images(image_paths=self.image_paths)

    def __display_images(self, image_paths, tags=[]):
        self.__clear_image_panel()
        image_grid = wx.FlexGridSizer(cols=10, hgap=10, vgap=10)
        all_images = list()
        for idx, name in enumerate(image_paths):
            bitmap = self.__scale_image(wx.Bitmap(name))
            if len(tags) > 0:
                image_name = name.split(os.path.sep)[-1]
                true_label = int(image_name.split('_')[-1][5])

                dc = wx.MemoryDC(bitmap)
                text = f'  P: {tags[idx]}\n  T: {Classifier.CLASSES[true_label]}'
                w, h = dc.GetSize()
                tw, th = dc.GetTextExtent(text)
                dc.DrawText(text, (w - tw) / 12, (h - th) / 2)  # display text in center

                del dc

            img_sbitmap = wx.StaticBitmap(self.image_panel, -1, bitmap)

            all_images.append(img_sbitmap)

        image_grid.AddMany(all_images)

        self.image_panel.SetSizerAndFit(image_grid)
        self.window.Layout()

    def __scale_image(self, bitmap, width=80, height=80):
        img = wx.ImageFromBitmap(bitmap)
        scaled_image = img.Scale(width, height, wx.IMAGE_QUALITY_HIGH)
        return wx.BitmapFromImage(scaled_image)

    def __import_files(self):
        wildcards = "All files (*.*) | *.*"

        file_dialog = wx.FileDialog(self, message="Choose a file",
                                    defaultDir=os.getcwd(),
                                    defaultFile="",
                                    wildcard=wildcards,
                                    style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_CHANGE_DIR)

        paths = []
        if file_dialog.ShowModal() == wx.ID_OK:
            paths = file_dialog.GetPaths()
            self.display_text_box.AppendText("You have selected " + str(len(paths)) + " files!\n")
            for path in paths:
                self.display_text_box.AppendText("File path: " + path + "\n")
        else:
            self.display_text_box.AppendText("File is not imported!\n")
            wx.MessageBox("Error: No files were imported!")

        self.display_text_box.AppendText("\n")

        return paths

    def __clear(self, event):
        self.display_text_box.Clear()

        self.__clear_image_paths()
        self.__clear_image_panel()

    def __clear_image_panel(self):
        self.image_panel.Freeze()
        self.image_panel.DestroyChildren()

        self.image_panel.Refresh()
        self.image_panel.Thaw()

    def __clear_image_paths(self):
        self.image_paths.clear()

    def __clear_weight_paths(self):
        self.weights_path.clear()

    def __clear_history_paths(self):
        self.history_path.clear()

    def __thread_start(self, event):
        thread = td.Thread(target=self.__run_training, args=(event, ))
        thread.start()











