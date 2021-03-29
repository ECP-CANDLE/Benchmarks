import json
import urllib
import requests
from tqdm import tqdm
import os


modac_user = None
modac_pass = None

def get_file_from_modac(fname, origin):
    """ Downloads a file from the "Model and Data Clearning House" (MoDAC)
    repository. Users should already have a MoDAC account to download the data.
    Accounts can be created on modac.cancer.gov

        Parameters
        ----------
        fname : string
            path on disk to save the file
        origin : string
            original MoDAC URL of the file

        Returns
        ----------
        string
            Path to the downloaded file      
    """
    print('Downloading data from modac.cancer.gov, make sure you have an account first.')
    total_size_in_bytes = get_dataObject_modac_filesize(origin)

    auth = authenticate_modac()
    data = json.dumps({})
    headers = {'Content-Type': 'application/json'}
    auth = (modac_user, modac_pass)

    post_url = origin + '/download'
    print("Downloading: " + post_url + " ...")
    response = requests.post(post_url, data = data, headers = headers, auth = auth, stream = True)
    if response.status_code != 200:
        print("Error downloading from modac.cancer.gov")
        raise Exception("Response code: {0}, Response message: {1}".format(response.status_code, response.text))
    block_size = 1024 #1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(fname, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        raise Exception ("ERROR, something went wrong while downloading ", post_url)

    print('Saved file to: ' + fname)
    return fname


def authenticate_modac(): 
    """
    Authenticates a user on modac.cancer.gov
        Returns
        ----------
        tuple(string,string) 
            tuple with the modac credentials
    """
 
    global modac_user
    global modac_pass
    if modac_user is None:
        modac_user = input("MoDaC Username: ")

    if modac_pass is None:
        import getpass
        modac_pass = getpass.getpass("MoDaC Password: ")

    return (modac_user, modac_pass)
    

def get_dataObject_modac_filesize(data_object_path):
    """
    Return the file size in bytes for a modac file 
        Parameters
        ----------
        data_object_path : string
            The path of the file on MoDAC

        Returns
        ----------
        integer
            file size in bytes  
    """
    self_dic = get_dataObject_modac_meta(data_object_path)
    if "source_file_size" in self_dic.keys():
        return int(self_dic["source_file_size"])
    else:
        return None

def get_dataObject_modac_md5sum(data_object_path):
    """
    Return the md5sum for a modac file 
        Parameters
        ----------
        data_object_path : string
            The path of the file on MoDAC

        Returns
        ----------
        string
            The md5sum of the file 
    """
    self_dic = get_dataObject_dme_meta(data_object_path)
    if "checksum" in self_dic.keys():
        return self_dic["checksum"]
    else:
        return None

def get_dataObject_modac_meta(data_object_path):
    """
    Return the self metadata values for a file (data_object)
        Parameters
        ----------
        data_object_path : string
            The path of the file on MoDAC

        Returns
        ----------
        dictionary
            Dictonary of all metadata for the file in MoDAC 
    """
    #data_object_path = encode_path(data_object_path)
    auth = authenticate_modac()

    get_response = requests.get(data_object_path, auth = auth)
    if get_response.status_code != 200:
        print("Error downloading from modac.cancer.gov", data_object_path)
        raise Exception("Response code: {0}, Response message: {1}".format(get_response.status_code, get_response.text))

    metadata_dic = json.loads(get_response.text)
    self_metadata = metadata_dic['metadataEntries']['selfMetadataEntries']['systemMetadataEntries']
    self_dic = {}
    for pair in self_metadata:
        self_dic[pair['attribute']] = pair['value'] 

    return self_dic
