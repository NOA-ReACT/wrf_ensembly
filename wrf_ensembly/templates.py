from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape

j2_env = Environment(
    loader=PackageLoader("wrf_ensembly", "template_files"),
    autoescape=select_autoescape(),
)


def generate(template_name: str, **kwargs) -> str:
    template = j2_env.get_template(template_name)
    return template.render(**kwargs)
