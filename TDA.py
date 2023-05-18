import pymatgen.core as pmcore
import pymatgen.io.cif as pmcif
import diode as dd
import dionysus as dy
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import re
import math
from persim import PersistenceImager
from persim.images_weights import linear_ramp
import persim
import glob

# Parameters for TDA calculations
abc_norm = 64.0  # Normalization factor for lattice vectors
exactness = True  # Whether to compute exact alpha complex
dim_homology = 2  # Dimension of homology to compute
pixel_size = 1.0  # Pixel size for persistence images
sigma = 0.15  # Sigma value for persistence images

# Get the folder path from the command line argument
folder_path = sys.argv[1]

number = 0

# Loop over files in the folder
for file_path in glob.glob(os.path.join(folder_path, "*")):
    xx = os.path.basename(file_path[4:])
    xx = xx[:xx.index("_")]
    MoleculeName = xx
    number = number + 1

    print(xx, number)

    # Read structure from CIF file
    struct = pmcore.Structure.from_file(file_path, primitive=False)
    aa = struct.lattice.a
    bb = struct.lattice.b
    cc = struct.lattice.c

    # Create a supercell of the structure
    n_a = round(abc_norm / aa)
    n_b = round(abc_norm / bb)
    n_c = round(abc_norm / cc)
    struct.make_supercell([n_a, n_b, n_c])

    sc_cif = pmcif.CifWriter(struct)

    points = struct.cart_coords

    # Generate alpha complex
    simplices = dd.fill_alpha_shapes(points, exact=exactness)

    # Convert alpha complex to filtration
    filtr = dy.Filtration(simplices)

    # Compute homology persistence
    ph = dy.homology_persistence(filtr)

    # Initialize persistence diagrams
    persdgm_all = dy.init_diagrams(ph, filtr)

    persdgm = np.empty(shape=(0, 2))

    # Extract birth and death coordinates from persistence diagrams
    for pt in persdgm_all[dim_homology]:
        r_birth = math.sqrt(pt.birth)
        r_death = math.sqrt(pt.death)
        if r_death <= 10000000000:
            persdgm = np.append(persdgm, np.array([[r_birth, r_death]]), axis=0)

    # Create persistence imager and set parameters
    persimg = PersistenceImager(pixel_size=pixel_size)
    persimg.fit(persdgm, skew=True)
    persimg.pers_range = (0, persimg.pers_range[1] + 5)
    persimg.birth_range = (0, persimg.birth_range[1] + 5)
    persimg.weight = linear_ramp
    persimg.weight_params = {'low': 1.0, 'high': 1.0, 'start': 0.0, 'end': 1.0}
    persimg.kernel_params = {'sigma': sigma}

    # Transform persistence diagrams to persistence images
    pimgs = persimg.transform(persdgm, skew=True)

    plt.axis('off')

    # Plot and save persistence image
    persimg.plot_image(pimgs)
    plt.tight_layout()
    plt.savefig('/path/{}.png'.format(str(MoleculeName)), bbox_inches='tight', pad_inches=0)

    # Break after processing the first CIF file
    break
