import numpy as np
from matplotlib import pyplot as plt

# Complete the following to make the plot
if __name__ == "__main__":
    data = np.load('sdss_galaxy_colors.npy')
    # Get a colour map
    cmap1 = plt.get_cmap('YlOrRd')

    # Define our colour indexes u-g and r-i
    cidx_ug = data['u'] - data['g']
    cidx_ri = data['r'] - data['i']
    
    # Make a redshift array
    redshift = data['redshift']

    # Create the plot with plt.scatter and plt.colorbar
    plt.scatter(cidx_ug, cidx_ri, c=redshift, lw=0, s=1, cmap=cmap1)
    colorbar = plt.colorbar().set_label("Redshift")
     
    # Define your axis labels and plot title
    plt.xlabel= "Colour index u-g"
    plt.ylabel= "Colour index r-i"
    plt.title = "Redshift (colour) u-g versus r-i"

    # Set any axis limits
    plt.axis([-0.5, 2.5, -0.5, 1])
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    #plt.legend()    
    plt.show()

