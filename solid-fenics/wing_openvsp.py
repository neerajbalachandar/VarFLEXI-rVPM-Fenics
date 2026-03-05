#import openvsp as vsp

#wing = vsp.AddGeom("WING")
'''
vsp.SetParmVal(wing,"Span","XSec_1",10)
vsp.SetParmVal(wing,"Root_Chord","XSec_1",1.5)
vsp.SetParmVal(wing,"Tip_Chord","XSec_1",0.5)

vsp.Update()
'''

import numpy as np
import pandas as pd

data = pd.read_csv("airfoil_DegenGeom.csv")
y = data["y"].values
x_le = data["xle"].values
z_le = data["zle"].values
x_te = data["xte"].values
z_te = data["zte"].values
c = data["chord"].values