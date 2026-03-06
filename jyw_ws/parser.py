import importlib
import inspect
import pkgutil
import sys
from argparse import ArgumentError
from collections.abc import Callable, Iterable, Sequence
from functools import wraps
from pathlib import Path
from pkgutil import ModuleInfo
from types import ModuleType
from typing import Any, TypeVar, cast
import draccus

F = TypeVar("F", bound=Callable[..., object])

PATH_KEY = "path"
PLUGIN_DISCOVERY_SUFFIX = "discover_packages_path"

def get_cli_overrides(field_name: str, args: Sequence[str] | None = None) -> list[str] | None:
    """Parses arguments from CLI at a given nested attribute level."""
    if args is None:
        args = sys.argv[1:]
    attr_level_args = []
    detect_string = f"--{field_name}."
    exclude_strings = (f"--{field_name}.{draccus.CHOICE_TYPE_KEY}=", f"--{field_name}.{PATH_KEY}=")
    for arg in args:
        if arg.startswith(detect_string) and not arg.startswith(exclude_strings):
            denested_arg = f"--{arg.removeprefix(detect_string)}"
            attr_level_args.append(denested_arg)
    return attr_level_args

def parse_arg(arg_name: str, args: Sequence[str] | None = None) -> str | None:
    if args is None:
        args = sys.argv[1:]
    prefix = f"--{arg_name}="
    for arg in args:
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None

def parse_plugin_args(plugin_arg_suffix: str, args: Sequence[str]) -> dict[str, str]:
    """Parse plugin-related arguments from command-line arguments."""
    plugin_args = {}
    for arg in args:
        if "=" in arg and plugin_arg_suffix in arg:
            key, value = arg.split("=", 1)
            if key.startswith("--"):
                key = key[2:]
            plugin_args[key] = value
    return plugin_args

class PluginLoadError(Exception):
    """Raised when a plugin fails to load."""

def load_plugin(plugin_path: str) -> None:
    """Load and initialize a plugin from a given Python package path."""
    try:
        package_module = importlib.import_module(plugin_path, __package__)
    except (ImportError, ModuleNotFoundError) as e:
        raise PluginLoadError(f"Failed to load plugin '{plugin_path}'. Verify the path and installation: {str(e)}") from e

    def iter_namespace(ns_pkg: ModuleType) -> Iterable[ModuleInfo]:
        return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")

    try:
        for _finder, pkg_name, _ispkg in iter_namespace(package_module):
            importlib.import_module(pkg_name)
    except ImportError as e:
        raise PluginLoadError(f"Failed to load plugin '{plugin_path}'. Verify the path and installation: {str(e)}") from e

def get_path_arg(field_name: str, args: Sequence[str] | None = None) -> str | None:
    return parse_arg(f"{field_name}.{PATH_KEY}", args)

def get_type_arg(field_name: str, args: Sequence[str] | None = None) -> str | None:
    return parse_arg(f"{field_name}.{draccus.CHOICE_TYPE_KEY}", args)

def filter_arg(field_to_filter: str, args: Sequence[str] | None = None) -> list[str]:
    if args is None:
        return []
    return [arg for arg in args if not arg.startswith(f"--{field_to_filter}=")]

def filter_path_args(fields_to_filter: str | list[str], args: Sequence[str] | None = None) -> list[str]:
    """Filters command-line arguments related to fields with specific path arguments."""
    if isinstance(fields_to_filter, str):
        fields_to_filter = [fields_to_filter]
    filtered_args = [] if args is None else list(args)

    for field in fields_to_filter:
        if get_path_arg(field, args):
            if get_type_arg(field, args):
                raise ArgumentError(
                    argument=None,
                    message=f"Cannot specify both --{field}.{PATH_KEY} and --{field}.{draccus.CHOICE_TYPE_KEY}",
                )
            filtered_args = [arg for arg in filtered_args if not arg.startswith(f"--{field}.")]

    return filtered_args

def wrap() -> Callable[[F], F]:
    """Handles additional processing for arguments like path arguments, plugins, and configuration loading."""

    def wrapper_outer(fn: F) -> F:
        @wraps(fn)
        def wrapper_inner(*args: Any, **kwargs: Any) -> Any:
            argspec = inspect.getfullargspec(fn)
            argtype = argspec.annotations[argspec.args[0]]
            if len(args) > 0 and type(args[0]) is argtype:
                cfg = args[0]
                args = args[1:]
            else:
                cli_args = sys.argv[1:]
                plugin_args = parse_plugin_args(PLUGIN_DISCOVERY_SUFFIX, cli_args)
                for plugin_cli_arg, plugin_path in plugin_args.items():
                    try:
                        load_plugin(plugin_path)
                    except PluginLoadError as e:
                        raise PluginLoadError(f"{e}\nFailed plugin CLI Arg: {plugin_cli_arg}") from e
                    cli_args = filter_arg(plugin_cli_arg, cli_args)

                # Now we just directly parse the configuration using draccus without any special handling for 'from_pretrained'
                cfg = draccus.parse(config_class=argtype, args=cli_args)
            response = fn(cfg, *args, **kwargs)
            return response

        return cast(F, wrapper_inner)

    return cast(Callable[[F], F], wrapper_outer)
