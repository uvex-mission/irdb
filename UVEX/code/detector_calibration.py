import numpy as np
from astropy.io import fits
from scopesim.detector import Detector

# UVEX detector dimensions
ny, nx = 4096, 4096


########## READ NOISE FUNCTIONS ###################


def generate_synthetic_read_noise(
    mean_read_noise,
    pixel_variation,
):
    """Generate a provisional per-pixel read-noise RMS map."""

    # Reproducible random number generator
    rng = np.random.default_rng(42)

    # Generate provisional per-pixel RMS values
    read_noise_map = rng.normal(
        loc=mean_read_noise,
        scale=pixel_variation,
        size=(ny, nx)
    )

    # RMS values cannot be negative
    read_noise_map = np.clip(read_noise_map, 0, None)

    # Store calibration map as float32
    read_noise_map = read_noise_map.astype(np.float32)

    return read_noise_map


def use_read_noise_array(
    measured_read_noise,
):
    """Use a measured per-pixel read-noise RMS array."""

    if measured_read_noise is None:
        raise ValueError(
            "No measured read-noise array was provided."
        )

    # Use the measured per-pixel RMS values
    read_noise_map = np.asarray(
        measured_read_noise,
        dtype=np.float32
    )

    # Check detector dimensions
    if read_noise_map.shape != (ny, nx):
        raise ValueError(
            f"Expected a {(ny, nx)} array, "
            f"but received {read_noise_map.shape}."
        )

    # Check for invalid values
    if not np.all(np.isfinite(read_noise_map)):
        raise ValueError(
            "Measured read-noise array contains NaN or infinite values."
        )

    if np.any(read_noise_map < 0):
        raise ValueError(
            "Measured read-noise RMS values cannot be negative."
        )

    # Store calibration map as float32
    read_noise_map = read_noise_map.astype(np.float32)

    return read_noise_map


def write_read_noise_calibration(
    read_noise_map,
    output_filename,
    detector_id,
    output_dir,
):
    """
    Write a measured per-pixel read-noise RMS map as a UVEX
    detector calibration product.

    Parameters
    ----------
    read_noise_map : ndarray
        2D array containing measured read-noise RMS values
        in e-/pixel/read.

    output_filename : str
        Output FITS filename.

    detector_id : str
        Identifier for the physical detector.

    output_dir : Path
        Directory where the calibration FITS file will be saved.
    """

    # Make sure we have the expected detector dimensions
    if read_noise_map.shape != (4096, 4096):
        raise ValueError(
            f"Expected (4096, 4096), got {read_noise_map.shape}"
        )

    # Make sure the calibration product contains valid RMS values
    if not np.all(np.isfinite(read_noise_map)):
        raise ValueError("Read-noise map contains NaN or infinite values.")

    if np.any(read_noise_map < 0):
        raise ValueError("Read-noise RMS cannot be negative.")

    # Store as float32
    read_noise_map = np.asarray(read_noise_map, dtype=np.float32)

    # Create FITS file
    hdu = fits.PrimaryHDU(read_noise_map)

    hdu.header["BUNIT"] = (
        "electron",
        "Read-noise RMS per pixel per read"
    )

    hdu.header["CALTYPE"] = (
        "READNOISE",
        "Detector calibration product"
    )

    hdu.header["DETSIZE"] = (
        "4096x4096",
        "Detector dimensions"
    )

    hdu.header["DETID"] = (
        detector_id,
        "Physical detector identifier"
    )

    hdu.header["STATUS"] = (
        "MEASURED",
        "Derived from detector characterization"
    )

    output_path = output_dir / output_filename

    hdu.writeto(output_path, overwrite=True)

    print(f"Saved:     {output_path}")
    print(f"Median RN: {np.median(read_noise_map):.3f} e-")
    print(f"Mean RN:   {np.mean(read_noise_map):.3f} e-")
    print(f"Min RN:    {np.min(read_noise_map):.3f} e-")
    print(f"Max RN:    {np.max(read_noise_map):.3f} e-")






########## DARK CURRENT FUNCTIONS ###################


def generate_synthetic_dark_current(
    mean_dark_current,
    pixel_variation,
):
    """Generate a provisional per-pixel dark-current map."""

    # Reproducible random number generator
    rng = np.random.default_rng(42)

    # Generate provisional per-pixel dark-current values
    dark_current_map = rng.normal(
        loc=mean_dark_current,
        scale=pixel_variation,
        size=(ny, nx)
    )

    # Dark-current values cannot be negative
    dark_current_map = np.clip(dark_current_map, 0, None)

    # Store calibration map as float32
    dark_current_map = dark_current_map.astype(np.float32)

    return dark_current_map


def use_dark_current_array(
    measured_dark_current,
):
    """Use a measured per-pixel dark-current array."""

    if measured_dark_current is None:
        raise ValueError(
            "No measured dark-current array was provided."
        )

    # Use the measured per-pixel dark-current values
    dark_current_map = np.asarray(
        measured_dark_current,
        dtype=np.float32
    )

    # Check detector dimensions
    if dark_current_map.shape != (ny, nx):
        raise ValueError(
            f"Expected a {(ny, nx)} array, "
            f"but received {dark_current_map.shape}."
        )

    # Check for invalid values
    if not np.all(np.isfinite(dark_current_map)):
        raise ValueError(
            "Measured dark-current array contains NaN or infinite values."
        )

    if np.any(dark_current_map < 0):
        raise ValueError(
            "Measured dark-current values cannot be negative."
        )

    # Store calibration map as float32
    dark_current_map = dark_current_map.astype(np.float32)

    return dark_current_map



def write_dark_current_calibration(
    dark_current_map,
    output_filename,
    detector_id,
    output_dir,
):
    """
    Write a per-pixel dark-current map as a UVEX
    detector calibration product.

    Parameters
    ----------
    dark_current_map : ndarray
        2D array containing dark-current values
        in e-/pixel/s.

    output_filename : str
        Output FITS filename.

    detector_id : str
        Identifier for the physical detector.

    output_dir : Path
        Directory where the calibration FITS file will be saved.
    """

    # Make sure we have the expected detector dimensions
    if dark_current_map.shape != (4096, 4096):
        raise ValueError(
            f"Expected (4096, 4096), got {dark_current_map.shape}"
        )

    # Make sure the calibration product contains valid values
    if not np.all(np.isfinite(dark_current_map)):
        raise ValueError(
            "Dark-current map contains NaN or infinite values."
        )

    if np.any(dark_current_map < 0):
        raise ValueError(
            "Dark-current values cannot be negative."
        )

    # Store as float32
    dark_current_map = np.asarray(
        dark_current_map,
        dtype=np.float32
    )

    # Create FITS file
    hdu = fits.PrimaryHDU(dark_current_map)

    hdu.header["BUNIT"] = (
        "electron/s",
        "Dark current per pixel"
    )

    hdu.header["CALTYPE"] = (
        "DARKCURRENT",
        "Detector calibration product"
    )

    hdu.header["DETSIZE"] = (
        "4096x4096",
        "Detector dimensions"
    )

    hdu.header["DETID"] = (
        detector_id,
        "Physical detector identifier"
    )

    hdu.header["STATUS"] = (
        "MEASURED",
        "Derived from detector characterization"
    )

    output_path = output_dir / output_filename

    hdu.writeto(output_path, overwrite=True)

    print(f"Saved:       {output_path}")
    print(f"Median DC:   {np.median(dark_current_map):.3f} e-/pixel/s")
    print(f"Mean DC:     {np.mean(dark_current_map):.3f} e-/pixel/s")
    print(f"Min DC:      {np.min(dark_current_map):.3f} e-/pixel/s")
    print(f"Max DC:      {np.max(dark_current_map):.3f} e-/pixel/s")







########## BIAS FUNCTIONS ###################



def generate_synthetic_bias(
    mean_bias,
    pixel_variation,
):
    """Generate a provisional per-pixel bias map."""

    # Reproducible random number generator
    rng = np.random.default_rng(42)

    # Generate provisional per-pixel bias values
    bias_map = rng.normal(
        loc=mean_bias,
        scale=pixel_variation,
        size=(ny, nx)
    )

    # Store calibration map as float32
    bias_map = bias_map.astype(np.float32)

    return bias_map


def use_bias_array(
    measured_bias,
):
    """Use a measured per-pixel bias array."""

    if measured_bias is None:
        raise ValueError(
            "No measured bias array was provided."
        )

    # Use the measured per-pixel bias values
    bias_map = np.asarray(
        measured_bias,
        dtype=np.float32
    )

    # Check detector dimensions
    if bias_map.shape != (ny, nx):
        raise ValueError(
            f"Expected a {(ny, nx)} array, "
            f"but received {bias_map.shape}."
        )

    # Check for invalid values
    if not np.all(np.isfinite(bias_map)):
        raise ValueError(
            "Measured bias array contains NaN or infinite values."
        )

    return bias_map





def write_bias_calibration(
    bias_map,
    output_filename,
    detector_id,
    output_dir,
):
    """
    Write a per-pixel bias map as a UVEX
    detector calibration product.
    """

    # Check detector dimensions
    if bias_map.shape != (4096, 4096):
        raise ValueError(
            f"Expected (4096, 4096), got {bias_map.shape}"
        )

    # Check for invalid values
    if not np.all(np.isfinite(bias_map)):
        raise ValueError(
            "Bias map contains NaN or infinite values."
        )

    # Store as float32
    bias_map = np.asarray(
        bias_map,
        dtype=np.float32
    )

    # Create FITS file
    hdu = fits.PrimaryHDU(bias_map)

    hdu.header["BUNIT"] = (
        "electron",
        "Bias level per pixel"
    )

    hdu.header["CALTYPE"] = (
        "BIAS",
        "Detector calibration product"
    )

    hdu.header["DETSIZE"] = (
        "4096x4096",
        "Detector dimensions"
    )

    hdu.header["DETID"] = (
        detector_id,
        "Physical detector identifier"
    )

    hdu.header["STATUS"] = (
        "MEASURED",
        "Derived from detector characterization"
    )

    output_path = output_dir / output_filename

    hdu.writeto(output_path, overwrite=True)

    print(f"Saved:       {output_path}")
    print(f"Median bias: {np.median(bias_map):.3f} e-")
    print(f"Mean bias:   {np.mean(bias_map):.3f} e-")
    print(f"Min bias:    {np.min(bias_map):.3f} e-")
    print(f"Max bias:    {np.max(bias_map):.3f} e-")











############ MISC FUNCTIONS ################

def create_blank_detector(detector_name):
    """Create a blank 4096 x 4096 UVEX detector for calibration tests."""

    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 4096
    header["NAXIS2"] = 4096

    detector = Detector(header)
    detector._hdu.header["NAME"] = detector_name

    return detector