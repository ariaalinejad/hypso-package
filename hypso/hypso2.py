from datetime import datetime
from dateutil import parser
from importlib.resources import files
from pathlib import Path
from typing import Literal, Union

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pyproj as prj
import xarray as xr

from .hypso import Hypso

from hypso.calibration import read_coeffs_from_file, \
                              run_radiometric_calibration, \
                              run_destriping_correction, \
                              run_smile_correction, \
                              make_mask, \
                              make_overexposed_mask, \
                              get_destriping_correction_matrix, \
                              run_destriping_correction_with_computed_matrix

from hypso.geometry import interpolate_at_frame, \
                           geometry_computation, \
                           get_nearest_pixel

from hypso.georeference import georeferencing
from hypso.georeference.utils import check_star_tracker_orientation

from hypso.reading import load_l1a_nc_cube, \
                          load_l1a_nc_metadata, \
                          load_l1b_nc_cube, \
                          load_l1b_nc_metadata #, \
                        #   load_l2a_nc_cube, \
                        #   load_l2a_nc_metadata

from hypso.writing import l1a_nc_writer, l1b_nc_writer, l2a_nc_writer

from satpy import Scene
from satpy.dataset.dataid import WavelengthRange

from pyresample.geometry import SwathDefinition

from trollsift import Parser

SUPPORTED_PRODUCT_LEVELS = ["l1a", "l1b", "l2a"]

ATM_CORR_PRODUCTS = ["6sv1", "acolite", "machi"]
CHL_EST_PRODUCTS = Literal["band_ratio", "6sv1_aqua", "acolite_aqua"]
LAND_MASK_PRODUCTS = Literal["global", "ndwi", "threshold"]
CLOUD_MASK_PRODUCTS = Literal["default"]

DEFAULT_ATM_CORR_PRODUCT = "6sv1"
DEFAULT_CHL_EST_PRODUCT = "band_ratio"
DEFAULT_LAND_MASK_PRODUCT = "global"
DEFAULT_CLOUD_MASK_PRODUCT = "default"

UNIX_TIME_OFFSET = 20 # TODO: Verify offset validity. Sivert had 20 here

# TODO: store latitude and longitude as xarray
# TODO: setattr, hasattr, getattr for setting class variables, especially those from geometry

class Hypso2(Hypso):

    def __init__(self, path: Union[str, Path], points_path: Union[str, Path, None] = None, verbose=False) -> None:
        
        """
        Initialization of (planned) HYPSO-2 Class.

        :param path: Absolute path to "L1a.nc" file
        :param points_path: Absolute path to the corresponding ".points" files generated with QGIS for manual geo
            referencing. (Optional. Default=None)

        """

        super().__init__(path=path) 

        # General --------------------------------------------------------
        self.platform = 'hypso2'
        self.sensor = 'hypso2_hsi'
        self.VERBOSE = verbose

        self._load_file(path=path)
        self._load_points_file(path=points_path)


        return None

    # Setters

    def _set_capture_type(self) -> None:
        """
        Format and set the capture region using information derived from the capture name.

        :return: None.
        """

        if self.capture_config["frame_count"] == self.standard_dimensions["nominal"]:

            self.capture_type = "nominal"

        elif self.image_height == self.standard_dimensions["wide"]:

            self.capture_type = "wide"
        else:
            # EXPERIMENTAL_FEATURES
            if self.VERBOSE:
                print("[WARNING] Number of Rows (AKA frame_count) Is Not Standard.")
            
            self.capture_type = "custom"

        if self.VERBOSE:
            print(f"[INFO] Capture capture type: {self.capture_type}")

        return None

    def _set_adcs_dataframes(self) -> None:

        self._set_adcs_pos_dataframe()
        self._set_adcs_quat_dataframe()

        return None
    
    # TODO move DataFrame formatting related code to geometry
    def _set_adcs_pos_dataframe(self) -> None:

        
        position_headers = ["timestamp", "eci x [m]", "eci y [m]", "eci z [m]"]
        
        timestamps = self.adcs["timestamps"]
        pos_x = self.adcs["position_x"]
        pos_y = self.adcs["position_y"]
        pos_z = self.adcs["position_z"]

        pos_array = np.column_stack((timestamps, pos_x, pos_y, pos_z))
        pos_df = pd.DataFrame(pos_array, columns=position_headers)

        self.adcs_pos_df = pos_df

        return None

    # TODO move DataFrame formatting related code to geometry
    def _set_adcs_quat_dataframe(self) -> None:

        
        quaternion_headers = ["timestamp", "quat_0", "quat_1", "quat_2", "quat_3", "Control error [deg]"]

        timestamps = self.adcs["timestamps"]
        quat_s = self.adcs["quaternion_s"]
        quat_x = self.adcs["quaternion_x"]
        quat_y = self.adcs["quaternion_y"]
        quat_z = self.adcs["quaternion_z"]
        control_error = self.adcs["control_error"]

        quat_array = np.column_stack((timestamps, quat_s, quat_x, quat_y, quat_z, control_error))
        quat_df = pd.DataFrame(quat_array, columns=quaternion_headers)

        self.adcs_quat_df = quat_df

        return None

    def _set_background_value(self) -> None:

        self.background_value = 8 * self.capture_config["bin_factor"] # ---------------------------------why is this 8 * bin factor? is that H1 specific?

        return None

    def _set_exposure(self) -> None:

        self.exposure = self.capture_config["exposure"] / 1000  # in seconds

        return None

    def _set_aoi(self) -> None:

        self.x_start = self.capture_config["aoi_x"]
        self.x_stop = self.capture_config["aoi_x"] + self.capture_config["column_count"]
        self.y_start = self.capture_config["aoi_y"]
        self.y_stop = self.capture_config["aoi_y"] + self.capture_config["row_count"]

    def _set_dimensions(self) -> None:

        self.bin_factor = self.capture_config["bin_factor"]

        self.row_count = self.capture_config["row_count"]
        self.frame_count = self.capture_config["frame_count"]
        self.column_count = self.capture_config["column_count"]

        self.image_height = self.capture_config["row_count"]
        self.image_width = int(self.capture_config["column_count"] / self.capture_config["bin_factor"])
        self.im_size = self.image_height * self.image_width

        self.bands = self.image_width
        self.lines = self.capture_config["frame_count"]  # AKA Frames AKA Rows
        self.samples = self.image_height  # AKA Cols

        self.spatial_dimensions = (self.capture_config["frame_count"], self.image_height)

        if self.VERBOSE:
            print(f"[INFO] Capture spatial dimensions: {self.spatial_dimensions}")

        return None

    def _set_timestamps(self) -> None:

        self.start_timestamp_capture = int(self.timing['capture_start_unix']) + UNIX_TIME_OFFSET

        # Get END_TIMESTAMP_CAPTURE
        # can't compute end timestamp using frame count and frame rate
        # assuming some default value if fps and exposure not available
        try:
            self.end_timestamp_capture = self.start_timestamp_capture + self.capture_config["frame_count"] / self.capture_config["fps"] + self.capture_config["exposure"] / 1000.0
        except:
            if self.VERBOSE:
                print("[WARNING] FPS or exposure values not found. Assuming 20.0 for each.")
            self.end_timestamp_capture = self.start_timestamp_capture + self.capture_config["frame_count"] / 20.0 + 20.0 / 1000.0

        # using 'awk' for floating point arithmetic ('expr' only support integer arithmetic): {printf \"%.2f\n\", 100/3}"
        time_margin_start = 641.0  # 70.0
        time_margin_end = 180.0  # 70.0

        self.start_timestamp_adcs = self.start_timestamp_capture - time_margin_start
        self.end_timestamp_adcs = self.end_timestamp_capture + time_margin_end

        self.unixtime = self.start_timestamp_capture
        self.iso_time = datetime.utcfromtimestamp(self.unixtime).isoformat()

        return None

    def _set_field_info(self, fields: dict) -> None:

        for key, value in fields.items():
            setattr(self, key, value) 

        return None
    
    def _set_capture_name(self, fields: dict) -> None:

        capture_name = self._compose_capture_name(fields=fields)

        setattr(self, "capture_name", capture_name)

        return None

    def _set_filenames(self, parent_directory: Path, fields: dict) -> None:

        capture_name = self._compose_capture_name(fields=fields)

        l1a_nc_file = Path(parent_directory, capture_name + "-l1a.nc")
        l1b_nc_file = Path(parent_directory, capture_name + "-l1b.nc")
        l2a_nc_file = Path(parent_directory, capture_name + "-l2a.nc")

        setattr(self, "l1a_nc_file", l1a_nc_file)
        setattr(self, "l1b_nc_file", l1b_nc_file)
        setattr(self, "l2a_nc_file", l2a_nc_file)

        return None
    
    def _set_dirnames(self, parent_directory: Path, fields: dict) -> None:

        capture_name = self._compose_capture_name(fields=fields)

        capture_dir = Path(parent_directory.absolute(), capture_name + "_tmp")
        parent_dir = Path(parent_directory.absolute())

        setattr(self, "capture_dir", capture_dir)
        setattr(self, "parent_dir", parent_dir)

        return None
    
    # Filename parsing and composing

    def _compose_capture_name(self, fields: dict) -> str:


        if hasattr(self, '_use_old_filename_format'):
            p = Parser("{capture_target}_{capture_datetime:%Y-%m-%d_%H%MZ}") # Old filename format
        else:
            p = Parser("{capture_target}_{capture_datetime:%Y-%m-%dT%H-%M-%SZ}") # New filename format

        capture_name = p.compose(fields)

        return capture_name
    
    def _parse_filename(self, path: str) -> dict:

        path = Path(path).absolute()
        field = None

        try:
            # New filename format
            #aegean_2024-08-22T08-41-46Z-l1a.nc
            p = Parser("{capture_target}_{capture_datetime:%Y-%m-%dT%H-%M-%SZ}-{product_level:3s}{atmospheric_correction:->}.{file_type}")
            fields = p.parse(str(path.name))
        except:
            # Old filename format
            setattr(self, '_use_old_filename_format', True)
            p = Parser("{capture_target}_{capture_datetime:%Y-%m-%d_%H%MZ}-{product_level:3s}{atmospheric_correction:->}.{file_type}")
            fields = p.parse(str(path.name))
        
        return fields

    # Loading funcitons

    def _load_file(self, path: Path) -> None: 
        
        path = Path(path).absolute()

        fields = self._parse_filename(path=path)

        self._set_field_info(fields=fields)
        self._set_capture_name(fields=fields)
        self._set_filenames(fields=fields, parent_directory=path.parent)
        self._set_dirnames(fields=fields, parent_directory=path.parent)

        match fields['product_level']:

            case "l1a":
                if self.VERBOSE: print('[INFO] Loading L1a capture ' + self.capture_name)
                self._load_l1a(path=path)
            case "l1b":
                if self.VERBOSE: print('[INFO] Loading L1b capture ' + self.capture_name)
                self._load_l1b(path=path)
            # case "l2a":
            #     if self.VERBOSE: print('[INFO] Loading L2a capture ' + self.capture_name) ----------put this back at some point
            #     self._load_l2a(path=path)
            case _:
                print("[ERROR] Unsupported product level.")
        return None

    # L1a functions

    def _load_l1a(self, path: Path) -> None:

        self._load_l1a_nc_metadata(path=path)
        self._load_l1a_nc_cube(path=path)

        return None

    def _load_l1a_nc_cube(self, path: Path) -> None:

        self.l1a_cube = load_l1a_nc_cube(nc_file_path=path)

        return None

    def _load_l1a_nc_metadata(self, path: Path) -> None:
        
        capture_config, \
        timing, \
        target_coords, \
        adcs, \
        dimensions, \
        navigation = load_l1a_nc_metadata(nc_file_path=path, hypso2=True) # TODO remove hypso2 flag later!
        
        setattr(self, "capture_config", capture_config)
        setattr(self, "timing", timing)
        setattr(self, "adcs", adcs)
        setattr(self, "navigation", navigation)

        self._set_background_value()
        self._set_exposure()
        self._set_aoi()
        self._set_dimensions()
        self._set_timestamps()
        self._set_capture_type()
        self._set_adcs_dataframes()

        return None
    
    # L1b functions

    def _load_l1b(self, path: Path) -> None:

        self._load_l1b_nc_metadata(path=path)
        self._load_l1b_nc_cube(path=path)

        return None

    def _load_l1b_nc_cube(self, path: Path) -> None:

        self.l1b_cube = load_l1b_nc_cube(nc_file_path=path)

        return None

    def _load_l1b_nc_metadata(self, path: Path) -> None:
        
        capture_config, \
        timing, \
        target_coords, \
        adcs, \
        dimensions, \
        navigation = load_l1b_nc_metadata(nc_file_path=path, hypso2=True) # TODO remove hypso2 flag later
        
        setattr(self, "capture_config", capture_config)
        setattr(self, "timing", timing)
        setattr(self, "adcs", adcs)
        setattr(self, "navigation", navigation)

        self._set_background_value()
        self._set_exposure()
        self._set_aoi()
        self._set_dimensions()
        self._set_timestamps()
        self._set_capture_type()
        self._set_adcs_dataframes()

        return None
    
    # ...

    # Georeferencing functions

    # TODO refactor
    def _load_points_file(self, 
                          path: str, 
                          image_mode: str = None, 
                          origin_mode: str = 'qgis',
                          flip_lats: bool = False,
                          flip_lons: bool = False) -> None:


        if path:
            path = Path(path).absolute()
        else:
            if self.VERBOSE:
                print('[INFO] No georeferencing .points file provided. Skipping georeferencing.')
            return None


        if not origin_mode:
            origin_mode = 'qgis'

        # Compute latitude and longitudes arrays if a points file is available

        if self.VERBOSE:
            print('[INFO] Running georeferencing...')

        gr = georeferencing.Georeferencer(filename=path,
                                            cube_height=self.spatial_dimensions[0],
                                            cube_width=self.spatial_dimensions[1],
                                            image_mode=image_mode,
                                            origin_mode=origin_mode)
        
        # Update latitude and longitude arrays with computed values from Georeferencer
        #self.latitudes = gr.latitudes[:, ::-1]
        #self.longitudes = gr.longitudes[:, ::-1]

        self.latitudes = gr.latitudes
        self.longitudes = gr.longitudes
        
        self._compute_flip()
        self._compute_bbox()
        self._compute_gsd()
        self._compute_resolution()

        if flip_lons:
            self.latitudes = self.latitudes[:,::-1]
            self.longitudes = self.longitudes[:,::-1]

        if flip_lats:
            self.latitudes = self.latitudes[::-1,:]
            self.longitudes = self.longitudes[::-1,:]


        self.latitudes_original = self.latitudes
        self.longitudes_original = self.longitudes

        self.georeferencing_has_run = True

        return None
    
    def _compute_flip(self) -> None:

        datacube_flipped = check_star_tracker_orientation(adcs_samples=self.adcs['adcssamples'],
                                                         quaternion_s=self.adcs['quaternion_s'],
                                                         quaternion_x=self.adcs['quaternion_x'],
                                                         quaternion_y=self.adcs['quaternion_y'],
                                                         quaternion_z=self.adcs['quaternion_z'],
                                                         velocity_x=self.adcs['velocity_x'],
                                                         velocity_y=self.adcs['velocity_y'],
                                                         velocity_z=self.adcs['velocity_z'])

        if not datacube_flipped:

            if self.l1a_cube is not None:
                self.l1a_cube = self.l1a_cube[:, ::-1, :]

            if self.l1b_cube is not None:  
                self.l1b_cube = self.l1b_cube[:, ::-1, :]
                
            if self.l2a_cube is not None:  
                self.l2a_cube = self.l2a_cube[:, ::-1, :]


        self.datacube_flipped = datacube_flipped

        return None
    
    def _compute_gsd(self) -> None:

        frame_count = self.frame_count
        image_height = self.image_height

        latitudes = self.latitudes
        longitudes = self.longitudes

        try:
            bbox = self.bbox
        except:
            self._compute_bbox()

        aoi = prj.aoi.AreaOfInterest(west_lon_degree=bbox[0],
                                     south_lat_degree=bbox[1],
                                     east_lon_degree=bbox[2],
                                     north_lat_degree=bbox[3], 
                                    )

        utm_crs_list = prj.database.query_utm_crs_info(datum_name="WGS 84", area_of_interest=aoi)


        #bbox_geodetic = [np.min(latitudes), 
        #                 np.max(latitudes), 
        #                 np.min(longitudes), 
        #                 np.max(longitudes)]

        #utm_crs_list = prj.database.query_utm_crs_info(datum_name="WGS 84",
        #                                                area_of_interest=prj.aoi.AreaOfInterest(
        #                                                west_lon_degree=bbox_geodetic[2],
        #                                                south_lat_degree=bbox_geodetic[0],
        #                                                east_lon_degree=bbox_geodetic[3],
        #                                                north_lat_degree=bbox_geodetic[1], )
        #                                            )
        
        if self.VERBOSE:
            print(f'[INFO] Using UTM map: ' + utm_crs_list[0].name, 'EPSG:', utm_crs_list[0].code)

        # crs_25832 = prj.CRS.from_epsg(25832) # UTM32N
        # crs_32717 = prj.CRS.from_epsg(32717) # UTM17S
        crs_4326 = prj.CRS.from_epsg(4326)  # Unprojected [(lat,lon), probably]
        source_crs = crs_4326
        destination_epsg = int(utm_crs_list[0].code)
        destination_crs = prj.CRS.from_epsg(destination_epsg)
        latlon_to_proj = prj.Transformer.from_crs(source_crs, destination_crs)


        pixel_coords_map = np.zeros([frame_count, image_height, 2])

        for i in range(frame_count):
            for j in range(image_height):
                pixel_coords_map[i, j, :] = latlon_to_proj.transform(latitudes[i, j], 
                                                                     longitudes[i, j])

        # time line x and y differences
        a = np.diff(pixel_coords_map[:, image_height // 2, 0])
        b = np.diff(pixel_coords_map[:, image_height // 2, 1])
        along_track_gsd = np.sqrt(a * a + b * b)
        along_track_mean_gsd = np.mean(along_track_gsd)

        # detector line x and y differences
        a = np.diff(pixel_coords_map[frame_count // 2, :, 0])
        b = np.diff(pixel_coords_map[frame_count // 2, :, 1])
        across_track_gsd = np.sqrt(a * a + b * b)
        across_track_mean_gsd = np.mean(across_track_gsd)


        self.along_track_gsd = along_track_gsd
        self.across_track_gsd = across_track_gsd

        self.along_track_mean_gsd = along_track_mean_gsd
        self.across_track_mean_gsd = across_track_mean_gsd

        return None

    def _compute_resolution(self) -> None:

        distances = [self.along_track_mean_gsd, 
                     self.across_track_mean_gsd]

        filtered_distances = [d for d in distances if d is not None]

        try:
            resolution = max(filtered_distances)
        except ValueError:
            resolution = 0

        self.resolution = resolution

        return None

    def _compute_bbox(self) -> None:

        lon_min = self.longitudes.min()
        lon_max = self.longitudes.max()
        lat_min = self.latitudes.min()
        lat_max = self.latitudes.max()

        bbox = (lon_min,lat_min,lon_max,lat_max)
        
        self.bbox = bbox

        return None


    # Calibration functions <---------------------------------------------------------
        
    def _run_calibration(self, 
                         overwrite: bool = False,
                         rad_cal = True,
                         smile_corr = True,
                         destriping_corr = False,
                         **kwargs) -> None:
        """
        Get calibrated and corrected cube. Includes Radiometric, Smile and Destriping Correction.
            Assumes all coefficients has been adjusted to the frame size (cropped and
            binned), and that the data cube contains 12-bit values.

        :return: None
        """

        if self.VERBOSE:
            print('[INFO] Running calibration routines...')

        self._set_calibration_coeff_files()
        self._set_calibration_coeffs()
        self._set_wavelengths()
        self._set_srf()

        l1a_cube = self.l1a_cube.to_numpy()

        if rad_cal:
            l1b_cube = self._run_radiometric_calibration(cube=l1a_cube)
        if smile_corr:
            l1b_cube = self._run_smile_correction(cube=l1b_cube)
        # if destriping_corr:
            # l1b_cube = self._run_destriping_correction(cube=l1b_cube, **kwargs)


        #l1b_cube = self._format_l1b_cube(data=l1b_cube)

        self.l1b_cube = l1b_cube

        return None

    def _run_radiometric_calibration(self, cube: np.ndarray) -> np.ndarray:

        # Radiometric calibration

        if self.VERBOSE:
            print("[INFO] Running radiometric calibration...")

        #cube = self._get_flipped_cube(cube=cube)

        cube = run_radiometric_calibration(cube=cube, 
                                           background_value=self.background_value,
                                           exp=self.exposure,
                                           image_height=self.image_height,
                                           image_width=self.image_width,
                                           frame_count=self.frame_count,
                                           rad_coeffs=self.rad_coeffs)

        #cube = self._get_flipped_cube(cube=cube)

        # TODO: The factor by 10 is to fix a bug in which the coeff have a factor of 10
        cube = cube / 10 #<------------------------------------------------------------- assuming this is not true for the H2 coeffs

        return cube

    def _run_smile_correction(self, cube: np.ndarray) -> np.ndarray:

        # Smile correction

        if self.VERBOSE:
            print("[INFO] Running smile correction...")

        #cube = self._get_flipped_cube(cube=cube)

        cube = run_smile_correction(cube=cube, 
                                    smile_coeffs=self.smile_coeffs)

        #cube = self._get_flipped_10cube(cube=cube)

        return cube

    def _run_destriping_correction(self, cube: np.ndarray, compute_destriping_matrix=False) -> np.ndarray:
        """
        Apply destriping correction to L1a datacube.

        :param cube: Radiometrically and smile corrected L1a datacube.

        :return: Destriped datacube.
        """

        if self.VERBOSE:
            print("[INFO] Running destriping correction...")

        #cube = self._get_flipped_cube(cube=cube)

        if compute_destriping_matrix:
            water_mask = make_mask(cube=cube, sat_val_scale=0.25)
            
            #overexposed_mask = make_overexposed_mask(cube=cube)
            #overexposed_mask = overexposed_mask.astype(bool)
            #self.water_mask = water_mask
            #self.overexposed_mask = overexposed_mask
            #mask = water_mask | ~overexposed_mask
            #mask = np.full(self.spatial_dimensions, False)
            #self.mask = mask

            self.destriping_coeffs = get_destriping_correction_matrix(cube, water_mask=water_mask)

            cube = run_destriping_correction_with_computed_matrix(cube=cube, 
                                         destriping_coeffs=self.destriping_coeffs)
            
        else:

            cube = run_destriping_correction(cube=cube, destriping_coeffs=self.destriping_coeffs[:,:])
            #cube = run_destriping_correction(cube=cube, destriping_coeffs=self.destriping_coeffs[:,::-1])
        
        #cube = self._get_flipped_cube(cube=cube)

        return cube

    def _set_calibration_coeffs(self) -> None:
        """
        Set the calibration coefficients included in the package. This includes radiometric,
        smile and destriping correction.

        :return: None.
        """
        
        self._set_radiometric_coeffs()
        self._set_smile_coeffs()
        self._set_destriping_coeffs()
        self._set_spectral_coeffs()
        
        return None

    def _set_radiometric_coeffs(self) -> None:
        """
        Set the radiometric calibration coefficients included in the package.

        :return: None.
        """

        self.rad_coeffs = read_coeffs_from_file(self.rad_coeff_file)

        return None

    def _set_smile_coeffs(self) -> None:
        """
        Set the smile calibration coefficients included in the package.

        :return: None.
        """

        self.smile_coeffs = read_coeffs_from_file(self.smile_coeff_file)

        return None
    
    def _set_destriping_coeffs(self) -> None:
        """
        Set the destriping calibration coefficients included in the package.

        :return: None.
        """

        self.destriping_coeffs = read_coeffs_from_file(self.destriping_coeff_file)

        return None
    
    def _set_spectral_coeffs(self) -> None:
        """
        Set the spectral calibration coefficients included in the package.

        :return: None.
        """

        self.spectral_coeffs = read_coeffs_from_file(self.spectral_coeff_file)

        return None

    def _set_calibration_coeff_files(self) -> None:
        """
        Set the absolute path for the calibration coefficients included in the package. This includes radiometric,
        smile and destriping correction.

        :return: None.
        """

        self._set_rad_coeff_file()
        self._set_smile_coeff_file()
        self._set_destriping_coeff_file()
        self._set_spectral_coeff_file()

        return None
    
    # ---------------------------------------- Needs more heavy editing from here on out

    def _set_rad_coeff_file(self, rad_coeff_file: Union[str, Path, None] = None) -> None:
        """
        Set the absolute path for the radiometric coefficients based on the detected capture type (wide, nominal, or custom).

        :param rad_coeff_file: Path to radiometric coefficients file (optional)

        :return: None.
        """

        if rad_coeff_file:
            self.rad_coeff_file = rad_coeff_file
            return None

        # TODO should this be adjusted based on capture type, as for hypso-1?
        match self.capture_type:

            case "custom":
                npz_file_radiometric = "h2_radiometric_calibration_matrix_full.npz"
            
            case "nominal":
                # TODO nominal matrix should probably also be constructed for hypso-2
                npz_file_radiometric = None

            case "wide":
                # TODO the correctness of the wide matrix should be confirmed
                npz_file_radiometric = "h2_radiometric_calibration_matrix_wide.npz" 
            
            case _:
                npz_file_radiometric = None


        if npz_file_radiometric:
            rad_coeff_file = files('hypso.calibration').joinpath(f'hypso2_data/{npz_file_radiometric}')
        else: 
            rad_coeff_file = None

        self.rad_coeff_file = rad_coeff_file

        return None

    def _set_smile_coeff_file(self, smile_coeff_file: Union[str, Path, None] = None) -> None:

        """
        Set the absolute path for the smile coefficients based on the detected capture type (wide, nominal, or custom).

        :param smile_coeff_file: Path to smile coefficients file (optional)

        :return: None.
        """

        if smile_coeff_file:
            self.smile_coeff_file = smile_coeff_file
            return None

        match self.capture_type:

            case "custom":
                #csv_file_smile = "spectral_calibration_matrix_HYPSO-1_full_v1.csv" 
                npz_file_smile = None

            case "nominal":
                #csv_file_smile = "smile_correction_matrix_HYPSO-1_nominal_v1.csv"
                npz_file_smile = None

            case "wide":
                #csv_file_smile = "smile_correction_matrix_HYPSO-1_wide_v1.csv"
                # npz_file_smile = "smile_correction_matrix_HYPSO-1_wide_v1.npz"
                npz_file_smile = "smile_correction_matrix_HYPSO-2_wide.npz"

            case _:
                npz_file_smile = None

        if npz_file_smile:
            smile_coeff_file = files('hypso.calibration').joinpath(f'hypso2_data/{npz_file_smile}')
        else:
            smile_coeff_file = npz_file_smile

        self.smile_coeff_file = smile_coeff_file

        return None

    def _set_destriping_coeff_file(self, destriping_coeff_file: Union[str, Path, None] = None) -> None:

        """
        Set the absolute path for the destriping coefficients based on the detected capture type (wide, nominal, or custom).

        :param destriping_coeff_file: Path to destriping coefficients file (optional)

        :return: None.
        """

        if destriping_coeff_file:
            self.destriping_coeff_file = destriping_coeff_file
            return None

        match self.capture_type:

            case "custom":
                #csv_file_destriping = None
                npz_file_destriping = None

            case "nominal":
                #csv_file_destriping = "destriping_matrix_HYPSO-1_nominal_v1.csv"
                npz_file_destriping = "destriping_matrix_HYPSO-1_nominal_v1.npz"

            case "wide":
                #csv_file_destriping = "destriping_matrix_HYPSO-1_wide_v1.csv"
                npz_file_destriping = "destriping_matrix_HYPSO-1_wide_v1.npz"

            case _:
                #csv_file_destriping = None
                npz_file_destriping = None

        if npz_file_destriping:
            destriping_coeff_file = files('hypso.calibration').joinpath(f'hypso1_data/{npz_file_destriping}') #--------------- TODO remember to update
        else:
            destriping_coeff_file = None

        self.destriping_coeff_file = destriping_coeff_file

        return None

    def _set_spectral_coeff_file(self, spectral_coeff_file: Union[str, Path, None] = None) -> None:
        """
        Set the absolute path for the spectral coefficients (wavelengths) based on the detected capture type (wide, nominal, or custom).

        :param spectral_coeff_file: Path to spectral coefficients file (optional)

        :return: None.
        """

        if spectral_coeff_file:
            self.spectral_coeff_file = spectral_coeff_file
            return None
        
        # TODO should one of the other two spectral calibration files be used?
        # npz_file_spectral = "h2_spectral_calibration_matrix.npz"
        # npz_file_spectral = "h2_spectral_calibration_wavelengths_center_row.npz"
        npz_file_spectral = "spectral_bands_HYPSO-2.npz"

        spectral_coeff_file = files('hypso.calibration').joinpath(f'hypso2_data/{npz_file_spectral}')

        self.spectral_coeff_file = spectral_coeff_file

        return None

    def _set_wavelengths(self) -> None:
        """
        Set the wavelengths corresponding to each of the 120 bands of the HYPSO datacube using information from the spectral coefficients.

        :return: None.
        """

        if self.spectral_coeffs is not None:
            self.wavelengths = self.spectral_coeffs
        else:
            self.wavelengths = range(0,120)

        return None

    # TODO SHOULD BE UPDATED BASED ON HYPSO-2 FWHM!
    # TODO: move to calibration module?
    def _set_srf(self) -> None:
        """
        Set Spectral Response Functions (SRF) from HYPSO for each of the 120 bands. Theoretical FWHM of 3.33nm is
        used to estimate Sigma for an assumed gaussian distribution of each SRF per band.

        :return: None.
        """

        # TODO figure out why this gave the unexpected error!
        if not any(self.wavelengths):
            self.srf = None

        fwhm_nm = 3.33
        sigma_nm = fwhm_nm / (2 * np.sqrt(2 * np.log(2)))

        srf = []
        for band in self.wavelengths:
            center_lambda_nm = band
            start_lambda_nm = np.round(center_lambda_nm - (3 * sigma_nm), 4)
            soft_end_lambda_nm = np.round(center_lambda_nm + (3 * sigma_nm), 4)

            srf_wl = [center_lambda_nm]
            lower_wl = []
            upper_wl = []
            for ele in self.wavelengths:
                if start_lambda_nm < ele < center_lambda_nm:
                    lower_wl.append(ele)
                elif center_lambda_nm < ele < soft_end_lambda_nm:
                    upper_wl.append(ele)

            # Make symmetric
            while len(lower_wl) > len(upper_wl):
                lower_wl.pop(0)
            while len(upper_wl) > len(lower_wl):
                upper_wl.pop(-1)

            srf_wl = lower_wl + srf_wl + upper_wl

            good_idx = [(True if ele in srf_wl else False) for ele in self.wavelengths]

            # Delta based on Hypso Sampling (Wavelengths)
            gx = None
            if len(srf_wl) == 1:
                gx = [0]
            else:
                gx = np.linspace(-3 * sigma_nm, 3 * sigma_nm, len(srf_wl))
            gaussian_srf = np.exp(
                -(gx / sigma_nm) ** 2 / 2)  # Not divided by the sum, because we want peak to 1.0

            # Get final wavelength and SRF
            srf_wl_single = self.wavelengths
            srf_single = np.zeros_like(srf_wl_single)
            srf_single[good_idx] = gaussian_srf

            srf.append([srf_wl_single, srf_single])

        self.srf = srf

        return None
    
    # Geometry computation functions

    def _run_geometry(self, overwrite: bool = False) -> None:

        if self.VERBOSE:
            print("[INFO] Running geometry computation...")

        framepose_data = interpolate_at_frame(adcs_pos_df=self.adcs_pos_df,
                                              adcs_quat_df=self.adcs_quat_df,
                                              timestamps_srv=self.timing['timestamps_srv'],
                                              frame_count=self.capture_config['frame_count'],
                                              fps=self.capture_config['fps'],
                                              exposure=self.capture_config['exposure'],
                                              verbose=self.VERBOSE
                                              )


        wkt_linestring_footprint, \
           prj_file_contents, \
           local_angles, \
           geometric_meta_info, \
           pixels_lat, \
           pixels_lon, \
           sun_azimuth, \
           sun_zenith, \
           sat_azimuth, \
           sat_zenith = geometry_computation(framepose_data=framepose_data,
                                             image_height=self.image_height,
                                             verbose=self.VERBOSE
                                             )

        self.framepose_df = framepose_data

        self.wkt_linestring_footprint = wkt_linestring_footprint
        self.prj_file_contents = prj_file_contents
        self.local_angles = local_angles
        self.geometric_meta_info = geometric_meta_info

        self.solar_zenith_angles = sun_zenith.reshape(self.spatial_dimensions)
        self.solar_azimuth_angles = sun_azimuth.reshape(self.spatial_dimensions)

        self.sat_zenith_angles = sat_zenith.reshape(self.spatial_dimensions)
        self.sat_azimuth_angles = sat_azimuth.reshape(self.spatial_dimensions)

        #self.lat_original = pixels_lat.reshape(self.spatial_dimensions)
        #self.lon_original = pixels_lon.reshape(self.spatial_dimensions)

        self.latitudes_original = pixels_lat.reshape(self.spatial_dimensions)
        self.longitudes_original = pixels_lon.reshape(self.spatial_dimensions)

        return None
    
    # ...

    # SatPy functions

    def _generate_satpy_scene(self) -> Scene:

        scene = Scene()

        latitudes, longitudes = self._generate_latlons()

        swath_def = SwathDefinition(lons=longitudes, lats=latitudes)

        latitude_attrs = {
                         'file_type': None,
                         'resolution': self.resolution,
                         'standard_name': 'latitude',
                         'units': 'degrees_north',
                         'start_time': self.capture_datetime,
                         'end_time': self.capture_datetime,
                         'modifiers': (),
                         'ancillary_variables': []
                         }

        longitude_attrs = {
                          'file_type': None,
                          'resolution': self.resolution,
                          'standard_name': 'longitude',
                          'units': 'degrees_east',
                          'start_time': self.capture_datetime,
                          'end_time': self.capture_datetime,
                          'modifiers': (),
                          'ancillary_variables': []
                          }

        #scene['latitude'] = latitudes
        #scene['latitude'].attrs.update(latitude_attrs)
        #scene['latitude'].attrs['area'] = swath_def

        #scene['longitude'] = longitudes
        #scene['longitude'].attrs.update(longitude_attrs)
        #scene['longitude'].attrs['area'] = swath_def

        return scene

    def _generate_latlons(self) -> tuple[xr.DataArray, xr.DataArray]:
        
        print(self.latitudes)
        print(self.dim_names_2d)
        latitudes = xr.DataArray(self.latitudes, dims=self.dim_names_2d)
        longitudes = xr.DataArray(self.longitudes, dims=self.dim_names_2d)

        return latitudes, longitudes
    
    def _generate_swath_definition(self) -> SwathDefinition:

        latitudes, longitudes = self._generate_latlons()
        swath_def = SwathDefinition(lons=longitudes, lats=latitudes)

        return swath_def
    
    def _generate_l1a_satpy_scene(self) -> Scene:

        scene = self._generate_satpy_scene()
        swath_def= self._generate_swath_definition()

        try:
            cube = self.l1a_cube
        except:
            return None

        attrs = {
                'file_type': None,
                'resolution': self.resolution,
                'name': None,
                'standard_name': cube.attrs['description'],
                'coordinates': ['latitude', 'longitude'],
                'units': cube.attrs['units'],
                'start_time': self.capture_datetime,
                'end_time': self.capture_datetime,
                'modifiers': (),
                'ancillary_variables': []
                }   

        wavelengths = range(0,120)

        for i, wl in enumerate(wavelengths):

            data = cube[:,:,i]

            data = data.reset_coords(drop=True)
                
            name = 'band_' + str(i+1)
            scene[name] = data
            #scene[name] = xr.DataArray(data, dims=self.dim_names_2d)
            scene[name].attrs.update(attrs)
            scene[name].attrs['wavelength'] = WavelengthRange(min=wl, central=wl, max=wl, unit="band")
            scene[name].attrs['band'] = i
            scene[name].attrs['area'] = swath_def

        return scene

    def _generate_l1b_satpy_scene(self) -> Scene:

        scene = self._generate_satpy_scene()
        swath_def= self._generate_swath_definition()

        try:
            cube = self.l1b_cube
            wavelengths = self.wavelengths
        except:
            return None

        attrs = {
                'file_type': None,
                'resolution': self.resolution,
                'name': None,
                'standard_name': cube.attrs['description'],
                'coordinates': ['latitude', 'longitude'],
                'units': cube.attrs['units'],
                'start_time': self.capture_datetime,
                'end_time': self.capture_datetime,
                'modifiers': (),
                'ancillary_variables': []
                }   

        for i, wl in enumerate(wavelengths):

            data = cube[:,:,i]
            
            data = data.reset_coords(drop=True)

            name = 'band_' + str(i+1)
            scene[name] = data
            #scene[name] = xr.DataArray(data, dims=self.dim_names_2d)
            scene[name].attrs.update(attrs)
            scene[name].attrs['wavelength'] = WavelengthRange(min=wl, central=wl, max=wl, unit="nm")
            scene[name].attrs['band'] = i
            scene[name].attrs['area'] = swath_def

        return scene
    
    def _generate_l2a_satpy_scene(self) -> Scene:

        scene = self._generate_satpy_scene()
        swath_def= self._generate_swath_definition()

        try:
            cube = self.l2a_cube
            wavelengths = self.wavelengths
        except:
            return None

        attrs = {
                'file_type': None,
                'resolution': self.resolution,
                'name': None,
                'standard_name': cube.attrs['description'],
                'coordinates': ['latitude', 'longitude'],
                'units': cube.attrs['units'],
                'start_time': self.capture_datetime,
                'end_time': self.capture_datetime,
                'modifiers': (),
                'ancillary_variables': []
                }   

        for i, wl in enumerate(wavelengths):

            data = cube[:,:,i]

            data = data.reset_coords(drop=True)

            name = 'band_' + str(i+1)
            scene[name] = data
            #scene[name] = xr.DataArray(data, dims=self.dim_names_2d)
            scene[name].attrs.update(attrs)
            scene[name].attrs['wavelength'] = WavelengthRange(min=wl, central=wl, max=wl, unit="nm")
            scene[name].attrs['band'] = i
            scene[name].attrs['area'] = swath_def

        return scene
    
    def _generate_toa_reflectance_satpy_scene(self) -> Scene:

        scene = self._generate_satpy_scene()
        swath_def= self._generate_swath_definition()

        try:
            cube = self.toa_reflectance_cube
            wavelengths = self.wavelengths
        except:
            return None

        attrs = {
                'file_type': None,
                'resolution': self.resolution,
                'name': None,
                'standard_name': cube.attrs['description'],
                'coordinates': ['latitude', 'longitude'],
                'units': cube.attrs['units'],
                'start_time': self.capture_datetime,
                'end_time': self.capture_datetime,
                'modifiers': (),
                'ancillary_variables': []
                }   

        for i, wl in enumerate(wavelengths):

            data = cube[:,:,i]

            data = data.reset_coords(drop=True)

            name = 'band_' + str(i+1)
            scene[name] = data
            #scene[name] = xr.DataArray(data, dims=self.dim_names_2d)
            scene[name].attrs.update(attrs)
            scene[name].attrs['wavelength'] = WavelengthRange(min=wl, central=wl, max=wl, unit="nm")
            scene[name].attrs['band'] = i
            scene[name].attrs['area'] = swath_def

        return scene

    # ... 

    # Public functions

    def load_file(self, path: str = None) -> None:

        if path:
            self._load_file(path=path)

        return None
    

    def load_points_file(self, path: str = None, image_mode=None, origin_mode=None, flip_lats=False, flip_lons=False) -> None:

        if path:
            self._load_points_file(path=path, image_mode=image_mode, origin_mode=origin_mode, flip_lats=flip_lats, flip_lons=flip_lons)

        return None
    

    # Public L1a methods

    def get_l1a_cube(self) -> xr.DataArray:

        return self.l1a_cube
    
    # Public L1b methods

    def generate_l1b_cube(self, **kwargs) -> None:

        self._run_calibration(**kwargs) 

        return None

    def get_l1b_cube(self) -> xr.DataArray:

        return self.l1b_cube
    
    def get_l1b_spectrum(self, 
                        latitude=None, 
                        longitude=None,
                        x: int = None,
                        y: int = None
                        ) -> tuple[np.ndarray, str]:

        if self.l1b_cube is None:
            return None
        
        if latitude is not None and longitude is not None:
            idx = get_nearest_pixel(target_latitude=latitude, 
                                    target_longitude=longitude,
                                    latitudes=self.latitudes,
                                    longitudes=self.longitudes)

        elif x is not None and y is not None:
            idx = (x,y)
            
        else:
            return None

        spectrum = self.l1b_cube[idx[0], idx[1], :]

        return spectrum

    def plot_l1b_spectrum(self, 
                        latitude=None, 
                        longitude=None,
                        x: int = None,
                        y: int = None,
                        save: bool = False
                        ) -> None:
        
        if latitude is not None and longitude is not None:
            idx = get_nearest_pixel(target_latitude=latitude, 
                                    target_longitude=longitude,
                                    latitudes=self.latitudes,
                                    longitudes=self.longitudes)

        elif x is not None and y is not None:
            idx = (x,y)

        else:
            return None

        spectrum = self.l1b_cube[idx[0], idx[1], :]
        bands = self.wavelengths
        units = spectrum.attrs["units"]

        output_file = Path(self.parent_dir, self.capture_name + '_l1a_plot.png')

        plt.figure(figsize=(10, 5))
        plt.plot(bands, spectrum)
        plt.ylabel(units)
        plt.xlabel("Wavelength (nm)")
        plt.title(f"L1b (lat, lon) --> (X, Y) : ({latitude}, {longitude}) --> ({idx[0]}, {idx[1]})")
        plt.grid(True)

        if save:
            # TODO: TypeError: imsave() missing 1 required positional argument: 'arr'
            plt.imsave(output_file)
        else:
            plt.show()

        return None

    def write_l1b_nc_file(self, overwrite: bool = False) -> None:

        if Path(self.l1b_nc_file).is_file() and not overwrite:

            if self.VERBOSE:
                print("[INFO] L1b NetCDF file has already been generated. Skipping.")

            return None

        l1b_nc_writer(satobj=self, 
                      dst_l1b_nc_file=self.l1b_nc_file, 
                      src_l1a_nc_file=self.l1a_nc_file)

        return None

    # TODO: run product through validator, add proprty
    # TODO: move code to atmospheric module
    # TODO: add set functions
    def _run_toa_reflectance(self) -> None:

        #self._run_calibration()
        #self._run_geometry()
        
        # Get Local variables
        srf = self.srf
        toa_radiance = self.l1b_cube.to_numpy()

        scene_date = parser.isoparse(self.iso_time)
        julian_day = scene_date.timetuple().tm_yday
        solar_zenith = self.solar_zenith_angles

        # Read Solar Data
        solar_data_path = str(files('hypso.atmospheric').joinpath("Solar_irradiance_Thuillier_2002.csv"))
        solar_df = pd.read_csv(solar_data_path)

        # Create new solar X with a new delta
        solar_array = np.array(solar_df)
        current_num = solar_array[0, 0]
        delta = 0.01
        new_solar_x = [solar_array[0, 0]]
        while current_num <= solar_array[-1, 0]:
            current_num = current_num + delta
            new_solar_x.append(current_num)

        # Interpolate for Y with original solar data
        new_solar_y = np.interp(new_solar_x, solar_array[:, 0], solar_array[:, 1])

        # Replace solar Dataframe
        solar_df = pd.DataFrame(np.column_stack((new_solar_x, new_solar_y)), columns=solar_df.columns)

        # Estimation of TOA Reflectance
        band_number = 0
        toa_reflectance = np.empty_like(toa_radiance)
        for single_wl, single_srf in srf:
            # Resample HYPSO SRF to new solar wavelength
            resamp_srf = np.interp(new_solar_x, single_wl, single_srf)
            weights_srf = resamp_srf / np.sum(resamp_srf)
            ESUN = np.sum(solar_df['mW/m2/nm'].values * weights_srf)  # units matche HYPSO from device.py

            # Earth-Sun distance (from day of year) using julian date
            # http://physics.stackexchange.com/questions/177949/earth-sun-distance-on-a-given-day-of-the-year
            distance_sun = 1 - 0.01672 * np.cos(0.9856 * (
                    julian_day - 4))

            # Get toa_reflectance
            solar_angle_correction = np.cos(np.radians(solar_zenith))
            multiplier = (ESUN * solar_angle_correction) / (np.pi * distance_sun ** 2)
            toa_reflectance[:, :, band_number] = toa_radiance[:, :, band_number] / multiplier

            band_number = band_number + 1

        self.toa_reflectance_cube = xr.DataArray(toa_reflectance, dims=("y", "x", "band"))
        #self.toa_reflectance_cube.attrs['units'] = "sr^-1"
        #self.toa_reflectance_cube.attrs['description'] = "Top of atmosphere (TOA) reflectance"

        return None
    
    # Public georeferencing functions

    # TODO
    def load_georeferencing(self, path: str) -> None:

        return None
    
    # TODO
    def generate_georeferencing(self) -> None:

        return None
    
    # TODO
    def get_ground_control_points(self) -> None:

        return None

    # TODO
    def write_georeferencing(self, path: str) -> None:

        return None
    

    # Public geometry functions

    # TODO
    def load_geometry(self, path: str) -> None:

        return None

    def generate_geometry(self) -> None:

        self._run_geometry()

        return None

    # TODO
    def write_geometry(self, path: str) -> None:

        return None


    # Public land mask methods

    # TODO
    def load_land_mask(self, path: str) -> None:

        return None

    def generate_land_mask(self, land_mask_name: LAND_MASK_PRODUCTS = DEFAULT_LAND_MASK_PRODUCT, **kwargs) -> None:

        self._run_land_mask(land_mask_name=land_mask_name, **kwargs)

        return None

    def get_land_mask(self) -> xr.DataArray:

        return self.land_mask

    # TODO
    def write_land_mask(self, path: str) -> None:

        return None


    # Public cloud mask methods

    # TODO
    def load_cloud_mask(self, path: str) -> None:

        return None

    def generate_cloud_mask(self, cloud_mask_name: CLOUD_MASK_PRODUCTS = DEFAULT_CLOUD_MASK_PRODUCT, **kwargs):

        self._run_cloud_mask(cloud_mask_name=cloud_mask_name, **kwargs)

        return None

    def get_cloud_mask(self) -> xr.DataArray:

        return self.cloud_mask
    
    # TODO
    def write_cloud_mask(self, path: str) -> None:

        return None


    # Public unified mask methods

    def get_unified_mask(self) -> xr.DataArray:

        return self.unified_mask


    # Public chlorophyll methods

    # TODO
    def load_chlorophyll_estimates(self, path: str) -> None:

        return None

    def generate_chlorophyll_estimates(self, 
                                       product_name: str = DEFAULT_CHL_EST_PRODUCT,
                                       model: Union[str, Path] = None,
                                       factor: float = 0.1
                                       ) -> None:

        self._run_chlorophyll_estimation(product_name=product_name, model=model, factor=factor)

    def get_chlorophyll_estimates(self, product_name: str = DEFAULT_CHL_EST_PRODUCT,
                                 ) -> np.ndarray:

        key = product_name.lower()

        return self.chl[key]


    # TODO
    def write_chlorophyll_estimates(self, path: str) -> None:

        return None
    


    # Public custom products methods
    
    # TODO
    def load_products(self, path: str) -> None:

        return None

    # TODO
    def write_products(self, path: str) -> None:

        return None



    # Public top of atmosphere (TOA) reflectance methods

    # TODO
    def load_toa_reflectance(self, path: str) -> None:
        
        return None

    def generate_toa_reflectance(self) -> None:

        self._run_toa_reflectance()

        return None

    def get_toa_reflectance(self) -> xr.DataArray:
        """
        Convert Top Of Atmosphere (TOA) Radiance to TOA Reflectance.

        :return: Array with TOA Reflectance.
        """

        return self.toa_reflectance_cube


    # TODO
    def write_toa_reflectance(self, path: str) -> None:
        
        return None


    def get_l1a_satpy_scene(self) -> Scene:

        return self._generate_l1a_satpy_scene()

    def get_l1b_satpy_scene(self) -> Scene:

        return self._generate_l1b_satpy_scene()

    def get_l2a_satpy_scene(self) -> Scene:

        return self._generate_l2a_satpy_scene()
    
    def get_toa_reflectance_satpy_scene(self) -> Scene:

        return self._generate_toa_reflectance_satpy_scene()

    def get_chlorophyll_estimates_satpy_scene(self) -> Scene:

        return self._generate_chlorophyll_satpy_scene()

    def get_products_satpy_scene(self) -> Scene:

        return self._generate_products_satpy_scene()

    def get_bbox(self) -> tuple:
        
        return self.bbox
    
    def get_closest_wavelength_index(self, wavelength: Union[float, int]) -> int:

        wavelengths = np.array(self.wavelengths)
        differences = np.abs(wavelengths - wavelength)
        closest_index = np.argmin(differences)

        return closest_index
