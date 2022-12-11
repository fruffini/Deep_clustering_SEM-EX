import pandas as pd
import click
from easydict import EasyDict


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option('--data_to_shuffle', default="/mimer/NOBACKUP/groups/snic2022-5-277/ltronchin/data/interim/claro/data.xlsx", help='Directory for the input file.', required = True, metavar='PATH')
@click.option('--saving_file', default="/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/data/interim/claro/data_claro.xlsx",help='Directory for the output file.', required = True, metavar='PATH')
@click.option('--frac', type=int, default=1)


def main(**kwargs):
    opt= EasyDict(**kwargs)
    # Excel file to shuffle
    data_to_shuffle = opt.data_to_shuffle
    data_ = pd.read_excel(opt.data_to_shuffle).sample(frac=1, random_state=1)
    # File shuffled to save
    saving_file = opt.saving_file
    with pd.ExcelWriter(saving_file) as writer:
        data_.to_excel(writer, sheet_name='Shuffled_data')
        print('Data shuffled with success')

main()