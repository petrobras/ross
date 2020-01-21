import os
import weasyprint
import ross
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
import bokeh


def generate_PDF():
    rotor = ross.rotor_example()
  
    TEMPLATE = 'report.html'
    CSS = 'report.css'    
    OUTPUT_FILENAME = 'my-report.pdf'
    
    ROOT = Path(os.path.dirname(ross.__file__))/'API_Report'
    ASSETS_DIR = ROOT/'assets'
    TEMPLAT_SRC = ROOT/'templates'
    CSS_SRC = ROOT/ 'static/css'
    OUTPUT_DIR = ROOT/'output'    
    
    env = Environment(loader=FileSystemLoader(str(TEMPLAT_SRC)))
    template = env.get_template(TEMPLATE)
    css = str(CSS_SRC/ CSS)
    
    bokeh_fig = rotor.plot_rotor()
    bokeh_fig.plot_width=500
    bokeh_fig.plot_height=400
    bokeh_fig.output_backend = 'svg'
    bokeh.io.export_svgs(bokeh_fig,filename = Path(ASSETS_DIR)/'plot.svg')
    template_vars = {'ASSETS_DIR':ASSETS_DIR,'ROTOR_NAME':'CENTRIFUGAL COMPRESSOR','ROTOR_ID':'0123456789'}
    
    rendered_string = template.render(template_vars)
    html = weasyprint.HTML(string=rendered_string)
    report = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    html.write_pdf(report, stylesheets=[css])
    print(f'Report generated in {OUTPUT_DIR}')

generate_PDF()