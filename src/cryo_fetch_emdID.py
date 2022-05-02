#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 14:41:16 2022

@author: Yoel Shkolnisky
"""

import ftplib 
import logging
import os
import gzip
import shutil

def cryo_fetch_emdID(emdID,mrc_filename,verbose=1):
    """
    cryo_fetch_emdID  Fetch a density map from EMDB.
    
    This function fetchs the map file (MRC format) with the given emdID 
    (integer) from EMDB. The MRC file is saved into the file mrc_filename, 
    which should contain the full path to the saved file. 
        
    """
       
    logging.basicConfig(level=logging.INFO,\
                        format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger()
    logger.disabled = False
    if verbose == 0 : logger.disabled = True  
        
    emdIDstr = str(emdID)
    
    # Connect FTP Server
    hostname =  "ftp.ebi.ac.uk"
    username = "anonymous"
    password = ""
    
    logger.info('Establishing an FTP connection with the EMD server ...')
    ftp_server = ftplib.FTP(hostname,username,password)
    logger.info('FTP connection was established.')
    
    # force UTF-8 encoding
    ftp_server.encoding = "utf-8"
    
    logger.info('Downloading the zipped density map ...')
    remote_dir = '/pub/databases/emdb/structures/EMD-%s/map/' %(emdIDstr)
    filename = 'emd_%s.map.gz' %(emdIDstr)
    remote_filename = remote_dir+filename
    output_dir = os.path.dirname(mrc_filename)    
    local_filename = os.path.join(output_dir,filename)
        
    # Write file in binary mode    
    with open(local_filename, "wb") as file:
        # Command for Downloading the file "RETR filename"
        ftp_server.retrbinary(f"RETR {remote_filename}", file.write)
    logger.info('The zipped density map was downloaded as %s.',local_filename)
    
        
    logger.info('Closing the FTP connection ...')
    ftp_server.quit()
    logger.info('The FTP connection was closed.')
          
       
    logger.info('Unzipping the downloaded file ...')
    # mrc_filename = (local_filename.removesuffix('.map.gz'))+".mrc"
    with gzip.open(local_filename, 'rb') as f_in:
        with open(mrc_filename, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)  
            
    # Delete gzip file and keep only MRC file.
    os.remove(local_filename)
    logger.info('Done. Save map to %s', mrc_filename)






