from jinja2 import Environment, PackageLoader, select_autoescape

j2_env = Environment(
    loader=PackageLoader("wrf_ensembly", "template_files"),
    autoescape=select_autoescape(),
)
