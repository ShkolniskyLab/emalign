#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 14:41:16 2022

@author: yaelharpaz1
"""


import numpy as np
from ftplib import FTP
import logging
import tempfile
import zipfile
from io import BytesIO
import os
import gzip
import shutil
import mrcfile

def cryo_fetch_emdID(emdID,verbose=1):
    """
    cryo_fetch_emdID  Fetch a density map from EMDB.
    
    This function fetchs the map file (MRC format) with the given emdID 
    (integer) from EMDB. The file is downloaded and unzipped.
    
    Returns the 3D structure from the retrived map file.
    
    """
    
    logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger()
    if verbose == 0 : logger.disabled = True  
    
    emdIDstr = str(emdID)
    ftpServer = "ftp.ebi.ac.uk"
    
    ftpAddress = '/pub/databases/emdb/structures/EMD-%s/map/' %(emdIDstr)
    filename = 'emd_%s.map.gz' %(emdIDstr)
       
    logger.info('Establishing an FTP connection with the EMD server ...')
    ngdc = FTP(ftpServer)
    ngdc.login()
    ngdc.cwd(ftpAddress)
    logger.info('FTP connection was established.')
      
    
    logger.info('Downloading the zipped density map ...')
    tmpdir = tempfile.mkdtemp()
    zip_fn = os.path.join(tmpdir, 'archive.zip')
    zip_obj = zipfile.ZipFile(zip_fn, 'w')
    
    ngdc.retrlines('LIST')
    flo = BytesIO()
    ngdc.retrbinary("RETR " + filename ,flo.write) 
    zip_obj.writestr(filename, flo.getvalue())
    
    zip_obj.close()
    
    logger.info('The zipped density map was downloaded to %s.',tmpdir)
    
    logger.info('Closing the FTP connection ...')
    ngdc.close()
    logger.info('The FTP connection was closed.')

    logger.info('Unzipping the downloaded file ...')
    with zipfile.ZipFile(zip_fn, 'r') as zip_ref:
        zip_ref.extractall(tmpdir)

    open(tmpdir + '/' + filename.removesuffix('.gz'), 'x')
    with gzip.open(tmpdir + '/' + filename, 'rb') as f_in:
        with open(tmpdir + '/' + filename.removesuffix('.gz'), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        
    logger.info('Unzipping was completed.')
    
        
    vol = np.ascontiguousarray(mrcfile.open(tmpdir+ '/' + filename.removesuffix('.gz')).data.T)
    
    return vol






