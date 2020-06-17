# coding: utf-8
import os


def run_disperse(imfile, cut, MSC=False):
    """
    Run Disperse on the chosen fits image (in argument).
    Input: image directory, image name (image has to be a fits), cut is the
          threshold (manually here), and MSC default is False change to True if
          you have already run it once and it is stored in .
    Return two files: a fits image and a ascii skeleton
    """

    if MSC:
        command_mse = 'mse ' + imfile + ' -outDir ' + os.path.dirname(imfile) + ' -loadMSC ' + imfile + '.MSC' + ' -cut ' + str(cut) + ' -periodicity 0 -upSkl > '+ os.path.dirname(imfile) + '/log_disperse_tmp'
        print('mse ran with : ',command_mse)
        os.system(command_mse)
    else:
        command_mse = 'mse ' + imfile + ' -outDir ' + os.path.dirname(imfile) + ' -cut ' + cut + ' -periodicity 0 -upSkl > '+ os.path.dirname(imfile) + '/log_disperse_tmp'
        print('mse ran with : ',command_mse)
        os.system(command_mse)


    with open(os.path.dirname(imfile) + '/log_disperse_tmp','r') as f:
        lines = f.readlines()

    last_line = lines[-1]
    if last_line == '*********** ALL DONE ************\n':
        fname_line = lines[-3]
        ccut = fname_line.split('file')[1].split('...')[0].split('.up.')[0].split('fits')[1]
    else:
        raise ValueError('mse did not work')


    command_toFITS = 'skelconv ' + imfile + ccut + '.up.NDskl -outDir ' + os.path.dirname(imfile) + ' -toFITS'

    command_ascii = 'skelconv ' + imfile + ccut + '.up.NDskl -outDir ' + os.path.dirname(imfile) + ' -to NDskl_ascii'

    print('')
    print('skelconv ran with : ',command_toFITS)
    os.system(command_toFITS)
    print('')
    print('skelconv ran with : ',command_ascii)
    os.system(command_ascii)



    return
