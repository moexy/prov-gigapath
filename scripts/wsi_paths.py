from pathlib import Path

WSI_SUFFIXES = (".svs", ".ndpi", ".mrxs", ".tif", ".tiff")


def slide_id_from_path(path: Path) -> str:
    return path.name[:-9] if path.name.lower().endswith(".ome.tiff") else path.stem
