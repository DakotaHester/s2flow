import rasterio as rio
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

    # Write back with LZW compression
    with rio.open(path, 'w', **profile) as dst:
        dst.write(data)


def parse_args():

    parser = argparse.ArgumentParser(description="Recompress GeoTIFFs with LZW compression.")
    parser.add_argument('path', type=str, help="Path to the GeoTIFF file to recompress.")
    return parser.parse_args()