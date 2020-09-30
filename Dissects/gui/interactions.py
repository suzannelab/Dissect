from PyQt5.QtWidgets import QFileDialog, QApplication


def find_file_name(directory='./', filter="tif file (*.tif)"):
    """ Get file path and name for open file later.

    Parameters
    ----------
    directory: str, init dictory to the roots.
    filter: str, possibility to filter file entry
    """
    directory = './'
    app = QApplication([directory])
    fname = QFileDialog.getOpenFileName(None,
                                        "Select a file...",
                                        directory,
                                        filter=filter)
    return fname[0]