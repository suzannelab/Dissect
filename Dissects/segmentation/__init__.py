import os

seg_dir = os.path.abspath(os.path.dirname(__file__))
seg_list = os.listdir(seg_dir)

__doc__ = """Available predefined datasets:""" + "\n".join(seg_list)

