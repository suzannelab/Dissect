import os

stores_dir = os.path.abspath(os.path.dirname(__file__))
stores_list = os.listdir(stores_dir)

__doc__ = """Available predefined datasets:""" + "\n".join(stores_list)

