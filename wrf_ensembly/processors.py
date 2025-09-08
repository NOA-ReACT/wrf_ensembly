"""
Plugin system for postprocessing data using DataProcessor pipeline.
"""

import importlib
import importlib.util
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import traceback
from typing import List

import numpy as np
import xarray as xr
import xwrf  # noqa: F401

from wrf_ensembly import external
from wrf_ensembly.config import Config, ProcessorConfig
from wrf_ensembly.console import logger


@dataclass
class ProcessingContext:
    """
    Context information passed to processors during pipeline execution.

    Contains metadata about the current processing operation including
    file paths, cycle information, member numbers, and other relevant data.
    """

    member: int
    """Ensemble member number (0-based)"""

    cycle: int
    """Cycle number"""

    input_file: Path
    """Path to the input file being processed"""

    output_file: Path
    """Path where the output file should be written"""

    config: Config
    """Experiment configuration object"""


class DataProcessor(ABC):
    """
    Abstract base class for data processors in the postprocessing pipeline.

    Each processor takes an xarray Dataset and returns a modified Dataset.
    Processors can be chained together to form a processing pipeline.
    """

    def __init__(self, **kwargs):
        """Initialize the processor with configuration parameters."""
        pass

    @abstractmethod
    def process(self, ds: xr.Dataset, context: ProcessingContext) -> xr.Dataset:
        """
        Process the input dataset and return the modified dataset.

        Args:
            ds: Input xarray Dataset
            context: Processing context containing metadata like file paths,
                    cycle information, member numbers, etc.

        Returns:
            Modified xarray Dataset
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a human-readable name for this processor."""
        pass


class XWRFPostProcessor(DataProcessor):
    """
    Processor that applies xWRF postprocessing operations as well as some other minor
    niceties. Specifically:

    - Computes air density
    - Applies XWRF postprocessing operations (destagger, diagnostics, etc)
    - Renames the time dimension to 't'
    - Renames the coordinates to standard names (longitude, latitude)
    - Fixes `coordianates` attributes for de-staggered variables
    - Removes unused variables like CLAT
    - Filters variables based on a configurable pattern (postprocess.variables_to_keep)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, ds: xr.Dataset, context: ProcessingContext) -> xr.Dataset:
        """Apply xWRF postprocessing operations."""
        logger.debug(f"Applying xWRF postprocessing with {self.name}")

        # Get model level thickness before destaggering
        geopotential = ds["PH"] + ds["PHB"]
        height = geopotential / 9.81
        model_level_thickness = np.diff(height, axis=1)  # This is a numpy array

        # Compute more diagnostics and destagger
        ds = ds.xwrf.postprocess().xwrf.destagger().compute()

        # Fix time dimension
        ds = ds.swap_dims({"Time": "t"})
        ds = ds.rename({"XTIME": "t"})
        ds = ds.set_xindex("t")
        ds = ds.drop_vars("Time")
        ds.t.attrs["standard_name"] = "time"
        ds.t.attrs["axis"] = "T"

        # Store level thickness
        ds["level_thickness"] = (("t", "z", "y", "x"), model_level_thickness)

        # Compute air density
        column_mass_per_area = (ds["MU"] + ds["MUB"]) / 9.81
        layer_mass_per_area = np.stack(
            [column_mass_per_area.values] * ds.sizes["z"], axis=1
        ) * (-ds["DNW"].values.reshape(-1, 1, 1))
        air_density = layer_mass_per_area / model_level_thickness
        ds["air_density"] = (("t", "z", "y", "x"), air_density)
        ds["air_density"].attrs = {
            "units": "kg m-3",
            "standard_name": "air_density",
        }

        # Rename XLONG and XLAT to longitude and latitude
        if "XLONG" in ds and "XLAT" in ds:
            ds = ds.rename({"XLONG": "longitude", "XLAT": "latitude"})
            ds.longitude.attrs["standard_name"] = "longitude"
            ds.latitude.attrs["standard_name"] = "latitude"
            ds.longitude.attrs["axis"] = "X"
            ds.latitude.attrs["axis"] = "Y"

        # Since the projection object is not serialisable, we need to drop it before saving
        ds = ds.drop_vars("wrf_projection")

        # Fix some attributes:
        # 1) remove grid_mapping because we drop `wrf_projection`
        # 2) for de-staggered variables, rename coordinates
        # 3) Remove CLAT from coordinates
        coordinate_mappings = {
            "XLONG_U": "longitude",
            "XLAT_U": "latitude",
            "XLONG_V": "longitude",
            "XLAT_V": "latitude",
            "XTIME": "t",
        }
        for var in ds.data_vars:
            if "grid_mapping" in ds[var].attrs:
                del ds[var].attrs["grid_mapping"]
            if "coordinates" in ds[var].encoding:
                coordinates: str = ds[var].encoding["coordinates"]
                for k, v in coordinate_mappings.items():
                    coordinates = coordinates.replace(k, v)
                coordinates = coordinates.replace("CLAT", "").strip()
                ds[var].encoding["coordinates"] = coordinates
        ds = ds.drop("CLAT")

        # Filter variables if needed, ensuring we keep time and vertical coordinates
        variables_to_keep = context.config.postprocess.variables_to_keep
        if variables_to_keep:
            patterns = [re.compile(v) for v in variables_to_keep]
            vars_to_drop = [
                v for v in ds.data_vars if not any(p.match(str(v)) for p in patterns)
            ]
            ds = ds.drop_vars(vars_to_drop)

        # Remove staggered dimensions
        ds = ds.drop_vars(
            ["XLAT_U", "XLONG_U", "XLAT_V", "XLONG_V", "x_stag", "y_stag", "z_stag"]
        )

        return ds

    @property
    def name(self) -> str:
        return "xWRF Post-processor"


class ScriptProcessor(DataProcessor):
    """
    Processor that applies external scripts to the dataset.
    """

    def __init__(self, script: str, **kwargs):
        super().__init__(**kwargs)
        self.script = script
        if "{in}" not in script or "{out}" not in script:
            raise ValueError(
                f"Script '{script}' does not contain the required placeholders {{in}} and {{out}}"
            )

    def process(self, ds: xr.Dataset, context: ProcessingContext) -> xr.Dataset:
        """Apply external script to the dataset."""
        logger.debug(f"Applying script processor: {self.script}")

        # Create temporary files for input and output
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Write input dataset to temporary file
            input_file = tmpdir / "input.nc"
            comp = dict(zlib=True, complevel=3)
            encoding = {var: comp for var in ds.data_vars}
            ds.to_netcdf(input_file, unlimited_dims=["Time"], encoding=encoding)

            # Create output file path
            output_file = tmpdir / "output.nc"

            # Replace placeholders with actual file paths
            cmd = self.script.replace("{in}", str(input_file))
            cmd = cmd.replace("{out}", str(output_file))

            # Replace additional context placeholders
            cmd = cmd.replace("{d}", str(context.member))
            cmd = cmd.replace("{c}", str(context.cycle))

            # Execute the script
            res = external.runc(cmd.split())
            if res.returncode != 0:
                logger.error(
                    f"Command {' '.join(res.command)} failed with return code {res.returncode} and output: {res.output}"
                )
                raise external.ExternalProcessFailed(res)

            # Read the processed dataset
            if not output_file.exists():
                raise RuntimeError(f"Script did not create output file: {output_file}")

            return xr.open_dataset(output_file)

    @property
    def name(self) -> str:
        return f"Script: {self.script}"


class ProcessorPipeline:
    """
    A pipeline of DataProcessor objects that processes datasets sequentially.
    """

    def __init__(self, processors: List[DataProcessor]):
        self.processors = processors

    def process(self, ds: xr.Dataset, context: ProcessingContext) -> xr.Dataset:
        """Process the dataset through all processors in the pipeline."""
        logger.info(f"Processing dataset through {len(self.processors)} processors")

        result = ds
        for i, processor in enumerate(self.processors):
            logger.debug(f"Step {i + 1}/{len(self.processors)}: {processor.name}")
            result = processor.process(result, context)

        return result

    def add_processor(self, processor: DataProcessor):
        """Add a processor to the end of the pipeline."""
        self.processors.append(processor)

    def insert_processor(self, index: int, processor: DataProcessor):
        """Insert a processor at a specific position in the pipeline."""
        self.processors.insert(index, processor)


def load_processor_from_string(processor_spec: str, **kwargs) -> DataProcessor:
    """
    Load a processor from a module path string or Python file.

    Args:
        processor_spec: String in one of these formats:
                       - "module.path:ClassName" (import from installed module)
                       - "/path/to/file.py:ClassName" (import from file)
                       - "/path/to/file.py" (import from file, use class with same name as file)
                       - Built-in processor name ("xwrf-post", "script")
        **kwargs: Additional keyword arguments to pass to the processor constructor

    Returns:
        DataProcessor instance

    Examples:
        load_processor_from_string("my_package.processors:MyProcessor")
        load_processor_from_string("/home/user/my_processor.py:MyProcessor")
        load_processor_from_string("/home/user/my_processor.py")  # Uses class MyProcessor
        load_processor_from_string("xwrf-post")
        load_processor_from_string("script", script="python my_script.py {in} {out}")
    """
    # Handle built-in processors
    if processor_spec == "script":
        return ScriptProcessor(**kwargs)

    # Handle file-based processors
    if ".py" in processor_spec:
        file_path = processor_spec
        class_name = None

        # Check if class name is specified after colon
        if ":" in processor_spec:
            file_path, class_name = processor_spec.split(":", 1)
        else:
            # Use file name (without .py) as class name
            class_name = Path(file_path).stem.replace("_", "").title()

        try:
            # Load module from file
            spec = importlib.util.spec_from_file_location(
                "dynamic_processor", file_path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load spec from file '{file_path}'")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            processor_class = getattr(module, class_name)

            if not hasattr(processor_class, "process"):
                raise ValueError(
                    f"Class {class_name} in {file_path} does not implement required 'process' method"
                )

            return processor_class(**kwargs)

        except FileNotFoundError:
            raise FileNotFoundError(f"Processor file not found: {file_path}")
        except AttributeError as e:
            raise AttributeError(
                f"Class '{class_name}' not found in file '{file_path}': {e}"
            )

    # Handle external processors (module.path:ClassName)
    if ":" not in processor_spec:
        raise ValueError(
            f"Processor spec '{processor_spec}' must be in format 'module.path:ClassName' or '/path/to/file.py[:ClassName]'"
        )

    module_path, class_name = processor_spec.split(":", 1)

    try:
        module = importlib.import_module(module_path)
        processor_class = getattr(module, class_name)

        if not issubclass(processor_class, DataProcessor):
            raise ValueError(
                f"Class {class_name} in {module_path} is not a subclass of DataProcessor"
            )

        return processor_class(**kwargs)

    except ImportError as e:
        raise ImportError(f"Could not import module '{module_path}': {e}")
    except AttributeError as e:
        raise AttributeError(
            f"Class '{class_name}' not found in module '{module_path}': {e}"
        )


def create_pipeline_from_config(config: List[ProcessorConfig]) -> ProcessorPipeline:
    """
    Create a processor pipeline from configuration.

    Args:
        config: List of processor configurations, each containing:
               - 'processor': processor specification string
               - Additional keyword arguments for the processor

    Returns:
        ProcessorPipeline instance

    Example:
        config = [
            {"processor": "script", "script": "python add_new_variable.py {in} {out}"},
            {"processor": "my_package.processors:CustomProcessor", "param1": "value1"}
            {"processor": "/path/to/file.py:MyProcessor", "param2": "value2"},
        ]
    """

    # Start with the xWRF post-processor as the default, always as the first step
    processors: list[DataProcessor] = [XWRFPostProcessor()]

    for proc_config in config:
        processor_spec = proc_config.processor
        processors.append(
            load_processor_from_string(processor_spec, **proc_config.params)
        )

    return ProcessorPipeline(processors)


def process_file_with_pipeline(args):
    """
    Process a single file through the processor pipeline.

    This function is defined at module level to be pickleable for ProcessPoolExecutor.

    Args:
        args: Tuple of (input_file, output_file, processor_configs, context)
    """
    input_file, output_file, processor_configs, context = args

    try:
        # Create pipeline from config
        pipeline = create_pipeline_from_config(processor_configs)

        # Load, process, and save dataset
        with xr.open_dataset(input_file) as ds:
            processed_ds = pipeline.process(ds, context)

            comp = dict(zlib=True, complevel=3)
            encoding = {var: comp for var in processed_ds.data_vars}
            processed_ds.to_netcdf(output_file, unlimited_dims=["t"], encoding=encoding)

        logger.debug(f"Successfully processed {input_file} -> {output_file}")

    except Exception as e:
        logger.error(f"Error processing {input_file}: {e} ({traceback.format_exc()})")
        raise
