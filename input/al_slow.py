## sample problem based on a monodirectional beam incident on 2.7 g/cm aluminum, 
# 4 groups, based on EPICS Al data (eyeball average xs)

# group boundaries [10, 1e2, 1e3, 1e4, 1e5] eV.
# Stopping power based on Fig 3.11 in Monte Carlo Transport of Electrons and Photons
# (1988).



number_density_al =  2.7*6.022e23/26.9815

s_bound = [0.05, 2.5, 7.0, 21, 1.0] * 2.7 *100

group_stopping_power = [40, 80, 200, 100] * 2.7 *100
group_el_scatter = [6e6, 5e7, 1e8, 1.5e9]*(number_density_al) * 1e-24
group_total_xs = [1.5e7, 1.0e8, 1e9, 2e9]*(number_density_al) * 1e-24