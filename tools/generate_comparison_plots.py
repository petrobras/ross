import sys
from pathlib import Path

import numpy as np
import plotly.io as pio
from jinja2 import Environment, FileSystemLoader

sys.path.append(str(Path(__file__).parent.parent))
import ross as rs

dir_path = Path(__file__).parent
rotors = {
    k: rs.rotor_assembly.Rotor.load((dir_path / "data" / k))
    for k in ["rotor_example", "c123701", "injection"]
}

env = Environment(loader=FileSystemLoader(dir_path / "plots/templates/"))
template = env.get_template("template.html")


# plot_rotor
kwargs = {
    rotor: pio.to_html(
        rotors[rotor].plot_rotor(), include_plotlyjs=False, full_html=False
    )
    for rotor in rotors
}
output = template.render(**kwargs)

with open(f"{dir_path}/plots/plot_rotor.html", "w") as f:
    f.write(output)

# plot_ucs
kwargs = {
    rotor: pio.to_html(
        rotors[rotor].run_ucs().plot(), include_plotlyjs=False, full_html=False
    )
    for rotor in rotors
}
output = template.render(**kwargs)

with open(f"{dir_path}/plots/plot_ucs.html", "w") as f:
    f.write(output)

    # plot_mode_2d
    kwargs = {
        rotor: pio.to_html(
            rotors[rotor].run_modal(speed=100).plot_mode_2d(0),
            include_plotlyjs=False,
            full_html=False,
        )
        for rotor in rotors
    }
    output = template.render(**kwargs)

    with open(f"{dir_path}/plots/plot_mode_2d.html", "w") as f:
        f.write(output)

# plot_mode_3d
kwargs = {
    rotor: pio.to_html(
        rotors[rotor].run_modal(speed=100).plot_mode_3d(0),
        include_plotlyjs=False,
        full_html=False,
    )
    for rotor in rotors
}
output = template.render(**kwargs)

with open(f"{dir_path}/plots/plot_mode_3d.html", "w") as f:
    f.write(output)

# unbalance.plot
kwargs = {
    rotor: pio.to_html(
        rotors[rotor]
        .run_unbalance_response(
            node=0,
            unbalance_magnitude=0.5,
            unbalance_phase=0,
            frequency=np.linspace(0, 2000),
        )
        .plot([(0, 0)]),
        include_plotlyjs=False,
        full_html=False,
    )
    for rotor in rotors
}
output = template.render(**kwargs)

with open(f"{dir_path}/plots/plot_unbalance.html", "w") as f:
    f.write(output)

# campbell.plot
kwargs = {
    rotor: pio.to_html(
        rotors[rotor].run_campbell(speed_range=np.linspace(0, 2000)).plot(),
        include_plotlyjs=False,
        full_html=False,
    )
    for rotor in rotors
}
output = template.render(**kwargs)

with open(f"{dir_path}/plots/plot_campbell.html", "w") as f:
    f.write(output)

# static.plot_deformation
kwargs = {
    rotor: pio.to_html(
        rotors[rotor].run_static().plot_deformation(),
        include_plotlyjs=False,
        full_html=False,
    )
    for rotor in rotors
}
output = template.render(**kwargs)

with open(f"{dir_path}/plots/plot_deformation.html", "w") as f:
    f.write(output)
