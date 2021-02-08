from src.DBExpander import DBExpand
import os

dir = 'SQL_Data/constant_setup'
files = os.listdir('SQL_Data/constant_setup')
for f in files:
    file = os.path.join(dir, f)
    DBExpand(file)
