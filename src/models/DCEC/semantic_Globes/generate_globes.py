# This script is needed to generate the validation dataset:
# Options: 1) Number of globes - Dimensions of globes - centers
#          2) Grid insides
import os
import click
import random
import globes_util
import sys
import xlsxwriter
from easydict import EasyDict
from util import util_general
import numpy as np
from PIL import Image
def midpoints(x):
    sl = ()
    for i in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x



sys.argv.extend(
        [
            '--max_globes', '5',
            '--radius_range', '[40,50]',
            '--N_patients', '200'
        ]
    )
@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option('--export_data_dir', help='Directory for output dataset', default='/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/src/models/DCEC/data', metavar='PATH')
@click.option('--dataset_name', help='Name of the output dataset',  type=str, default='GLOBES')
@click.option('--N_patients', help='Number of patients to generate',  type=int, default=10)
@click.option('--max_globes', help='Number of internal globes inside the 3D images',  type=int, default=2)
@click.option('--image_dim', help='3DImage dimensions, (img_dim, img_dim, img_dim) ',  type=int, default=224)
@click.option('--radius_range',cls=util_general.ConvertStrToList, help='Radius intervals values',default=[])
@click.option('--volume',  is_flag=True )
@click.option('--surface_thickness', help='Number of pixels of surfaces\' thickness',  type=int, default=3)
@click.option('--margin', help='Number of pixels of margin from the image borders',  type=int, default=5)
@click.option('--sovrapposition', help='Max sovrapposition between spheres.', type=int, default=40)
@click.option('--step_slicing', help='Slicing pixels steps.', type=int, default=2)

def main(**kwargs):
    opt = EasyDict(**kwargs)

    # DEFINITION OF THE INTERNAL CUBE SPACE WHERE ARE GENERATED THE SPHERE CENTERS
    image_dim = opt.image_dim
    radius_range = opt.radius_range
    margin = opt.margin
    opt.space_dim = image_dim - (2 * radius_range[-1]) - 2 * margin
    # GENERATION OF FOLDERS and FILES
    # ------------------------------------------------------------------------------------------
    dataset_dir = os.path.join(opt.export_data_dir, opt.dataset_name)
    raw_dir = os.path.join(dataset_dir, 'raw')
    interim_dir =  os.path.join(dataset_dir, 'interim')
    images_dir = os.path.join(raw_dir, 'images')
    util_general.del_dir(interim_dir)
    util_general.del_dir(images_dir)
    util_general.mkdirs([interim_dir, images_dir])

    # Create a workbook and add a worksheet.

    workbook = xlsxwriter.Workbook(os.path.join(interim_dir,'data.xlsx'))
    worksheet = workbook.add_worksheet()

    # Add a bold format to use to highlight cells.
    bold = workbook.add_format({'bold': True})

    # Add a number format for cells with money

    # Write some data headers.
    worksheet.write('A1', 'img ID', bold)
    worksheet.write('B1', 'Patient ID', bold)
    worksheet.write('C1', 'Label: Number of Globes', bold)

    # Start from the first cell below the headers.
    row = 1
    col = 0






    # Generation of the patient ID and folder
    # ------------------------------------------------------------------------------------------
    # GENERATION OF SPHERES VARIABLES: (RADIUS, CENTER)

    for patient in range(1,opt.n_patients+1):
        IDENTIFIER = 'ID{0}'.format("{0:04d}".format(patient))
        patient_dir = os.path.join(images_dir, IDENTIFIER)
        util_general.del_dir(patient_dir)
        util_general.mkdir(patient_dir)
        print("GENERATION OF GLOBES FOR PATIENT: ", IDENTIFIER)
        # GENERATION OF GLOBES
        labels_number_globes = random.randint(2, opt.max_globes)
        spheres = globes_util.create_spheres_limited_sovrapposition(opt=opt, number_of_spheres=labels_number_globes)
        Volumetric_image = np.zeros((image_dim,) * 3)
        for sphere in spheres.values():
            radius = sphere[0]
            center = tuple(sphere[1])
            sphere = globes_util.sphere((image_dim,) * 3, radius, center)
            if not opt.volume:
                sphere -= globes_util.sphere((image_dim,) * 3, radius - opt.surface_thickness, center)
            Volumetric_image += sphere
        Volumetric_image[Volumetric_image >= 1] = 1

        # GENERATION OF SLICES
        z_ = [ sphere[1][2] for sphere in spheres.values()]
        rads_ = [sphere[0] for sphere in spheres.values()]
        min_z = min(z_)
        max_z = max(z_)
        bottom_limit = rads_[z_.index(min_z)]
        upper_limit = rads_[z_.index(max_z)]
        range_of_cut = range(min_z - int(np.round((3 / 4) * bottom_limit)), max_z + int(np.round((3 / 4) * upper_limit)), opt.step_slicing)
        for i, plane_k in enumerate(range_of_cut):
            patient_ID = IDENTIFIER+'_{0}'.format(str(i+1))
            plane = Volumetric_image[plane_k]
            img = Image.fromarray(plane)
            img.save(
                fp=os.path.join(patient_dir, "{0}.tif".format(patient_ID))
            )
            worksheet.write(row, 0, patient_ID)
            worksheet.write(row, 1, IDENTIFIER)
            worksheet.write(row, 2, labels_number_globes)
            row += 1

    print("finished")
    workbook.close()

if __name__ == '__main__':
    main()



