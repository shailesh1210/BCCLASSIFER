import wx
from ui.interface import Interface

def main():

    app = wx.App()

    ui = Interface()
    ui.initialize()

    app.MainLoop()


if __name__ == "__main__":
    main()

