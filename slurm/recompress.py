import rasterio as rio
import shutil
import argparse


def main():
    args = parse_args()
    recompress(args.path)


def recompress(path):
    
    with rio.open(path) as src:
        profile = src.profile.copy()
        data = src.read()

    # Update profile for LZW compression
    profile.update(
        compress='LZW',
        predictor=2,
        interleave='band',
        tiled=False,
        blockysize=2048,
    )
    profile.pop('blockxsize', None)  # Remove blockxsize if it exists
    
    temp_path = path + '.tmp'
    with rio.open(temp_path, 'w', **profile) as dst:
        dst.write(data)
    
    shutil.move(temp_path, path)  # Replace original file with recompressed version


def parse_args():

    parser = argparse.ArgumentParser(description="Recompress GeoTIFFs with LZW compression.")
    parser.add_argument('path', type=str, help="Path to the GeoTIFF file to recompress.")
    return parser.parse_args()


if __name__ == "__main__":
    main()