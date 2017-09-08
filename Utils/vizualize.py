#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:53:32 2017

@author: roberto
"""

import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#plotando longitude e latitude 
# alpha=0.1 me dá maior densidade dos dados

housing.plot(kind="scatter",x="longitude",y="latitude", alpha=0.1)

#plotando agora tb a densidade populacional , um circulo, opcao s
# assim como o preco , usando mapa de cor, , opcao cmap
 
housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,s=housing["population"]/100, label="population", figsize=(10,7),c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True,)
plt.legend()