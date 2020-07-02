import os

import ross.report as api

report = api.report_example()
D = [0.35, 0.35]
H = [0.08, 0.08]
HP = [10000, 10000]
RHO_ratio = [1.11, 1.14]
RHOd = 30.45
RHOs = 37.65
oper_speed = 1000.0

s = report.export_html( D=D, H=H, HP=HP, oper_speed=oper_speed, RHO_ratio=RHO_ratio, RHOs=RHOs,
                        RHOd=RHOd, output_path=os.getcwd())

# assets = report.assets_prep( D, H, HP, oper_speed, RHO_ratio, RHOs, RHOd)
# s = report.generate_report( D=D, H=H, HP=HP, oper_speed=oper_speed, RHO_ratio=RHO_ratio, RHOs=RHOs,
#                         RHOd=RHOd)
#
# s.run_server(debug=True)
