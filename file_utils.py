import sys
import pathlib

def get_cc359_filenames(base_folder):
    original_folder = base_folder.joinpath("cc359/Original")
    mask_folder = base_folder.joinpath("cc359/Silver-standard")
    files = []
    for original_filename in list(original_folder.glob("*.nii.gz")):
        mask_filename = mask_folder.joinpath(
            "{}_ss.nii.gz".format(original_filename.stem.split(".")[0])
        )
        files.append((original_filename, mask_filename))
    return files


def get_nfbs_filenames(base_folder):
    nfbs_folder = base_folder.joinpath("nfbs/NFBS_Dataset")
    files = []
    for original_filename in list(nfbs_folder.glob("**/*_T1w.nii.gz")):
        mask_filename = original_filename.parent.joinpath(
            "{}_brainmask.nii.gz".format(original_filename.stem.split(".")[0])
        )
        assert mask_filename.exists()
        files.append((original_filename, mask_filename))
    return files


def get_lidc_filenames(base_folder):
    """
    lung nodules dataset
    """
    
    lidc_folder = base_folder.joinpath("LIDC-IDRI")

    files = []

    for folder_name in list(lidc_folder.glob("*")):
        
        volume, mask = (folder_name.joinpath("ct.nii.gz"),\
                folder_name.joinpath("mask.nii.gz"))

        files.append((volume, mask))

    return files


def main():
    base_folder = pathlib.Path(sys.argv[1]).resolve()
    cc359_files = get_cc359_filenames(base_folder)
    nfbs_files = get_nfbs_filenames(base_folder)

    print(len(cc359_files))
    print(len(nfbs_files))

if __name__ == "__main__":
    main()
