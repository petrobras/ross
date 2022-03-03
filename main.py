from ross.fluid_flow.cylindrical import cylindrical_bearing_example

x0 = [0.1, -0.1]
bearing = cylindrical_bearing_example()
bearing.run(x0)
bearing.coefficients()
print(bearing.coefficients())
