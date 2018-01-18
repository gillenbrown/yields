import os
from collections import defaultdict

from scipy import interpolate
import numpy as np

iwamoto_file = "iwamoto_99_Ia_yields.txt"

nomoto_file = "nomoto_06_imf_weighted_II.txt"

my_nomoto_ave_file = "nomoto_06_imf_weighted_II_my_ave.txt"
my_nomoto_hn_file = "nomoto_06_imf_weighted_II_my_hn.txt"
my_nomoto_reg_file = "nomoto_06_imf_weighted_II_my_no_hn.txt"
my_ww_file = "ww_95_imf_weighted_II.txt"

nomoto_ind_0 = "nomoto_individual/z_0.txt"
nomoto_ind_0_001 = "nomoto_individual/z_0.001.txt"
nomoto_ind_0_004 = "nomoto_individual/z_0.004.txt"
nomoto_ind_0_02 = "nomoto_individual/z_0.02.txt"

nomoto_ind_0_hn = "nomoto_individual/z_0_hn.txt"
nomoto_ind_0_001_hn = "nomoto_individual/z_0.001_hn.txt"
nomoto_ind_0_004_hn = "nomoto_individual/z_0.004_hn.txt"
nomoto_ind_0_02_hn = "nomoto_individual/z_0.02_hn.txt"

ww_ind_sol_a = "ww_individual/ww95_5a.txt"
ww_ind_sol_b = "ww_individual/ww95_5b.txt"
ww_ind_0_1_sol_a = "ww_individual/ww95_10a.txt"
ww_ind_0_1_sol_b = "ww_individual/ww95_10b.txt"
ww_ind_0_01_sol_a = "ww_individual/ww95_12a.txt"
ww_ind_0_01_sol_b = "ww_individual/ww95_12b.txt"
ww_ind_4_sol_a = "ww_individual/ww95_14a.txt"
ww_ind_4_sol_b = "ww_individual/ww95_14b.txt"
ww_ind_0_a = "ww_individual/ww95_16a.txt"
ww_ind_0_b = "ww_individual/ww95_16b.txt"

def _get_data_path(data_file):
    """Returns the path of the Iwamoto input file on this machine.
    
    We know the relative path of it compared to this file, so it's easy to
    know where it is."""
    this_file_dir = os.path.dirname(__file__)
    return this_file_dir + "/data/{}".format(data_file)



def _parse_iwamoto_element(original_string):
    """Parses the LaTeX formatted string into an element that the code can use

    The original format is like "^{12}C". To parse this, we just find the 
    location of the brackets, then use that to know where the number and name
    are. This holds no matter how long the elemental names or numbers are"""

    first_bracket = original_string.index("{")
    second_bracket = original_string.index("}")
    # then use those locations to get the actual values we need.
    number = original_string[first_bracket + 1:second_bracket]
    name = original_string[second_bracket + 1:]
    return "{}_{}".format(name, number)

def _parse_nomoto_element(number, name):
    """The Nomoto 2006 file has a separate column for the name and the 
    mass number, so we can take those and turn them into one thing like
    the code wants. One minor hassle is that the file uses "p" and "d" for
    Hydrogen and Deuterium, respectively."""
    if number == "01" and name == "p":
        return "H_1"
    elif number == "02" and name == "d":
        return "H_2"
    else:
        return "{}_{}".format(name, number.lstrip("0"))

def _parse_nomoto_individual_element(name):
    """The file has elements in the format NumberName, like 9Be or 22Na"""
    if name == "p":
        return "H_1"
    elif name == "d":
        return "H_2"
    else:
        # we have to know where to put the underscore
        if name[1].isalpha():
            num = name[0]
            sym = name[1:]
        else:
            num = name[0:2]
            sym = name[2:]
        return "{}_{}".format(sym, num)


def _parse_iwamoto_model(full_name):
    """Parses the full name to get the needed model.
    
    :param full_name: Name of the model, in the format "iwamoto_99_Ia_MODEL" 
                      The valid names for MODEL are "W7", "W70", "WDD1", 
                      "WDD2", "WDD3", "CDD1", and "CDD2". 
    :returns: name of the model being used
    :rtype: str
    """
    # first check that the beginning is what we want
    if full_name[:14] != "iwamoto_99_Ia_":
        raise ValueError("This is not an Iwamoto model.")

    # we can then get the portion that is the model name
    model_name = full_name.split("_")[-1]

    # then check that it is indeed the right model
    acceptable_models = ["W7", "W70", "WDD1", "WDD2", "WDD3", "CDD1", "CDD2"]
    if model_name in acceptable_models:
        return model_name
    else:
        raise ValueError("Model supplied is not a valid Iwamoto 99 Ia model.")


def _metallicity_log(value):
    """When taking logs of metallicity, there is often a zero value that we
    don't want to break our code. I can assign a default value to that."""
    # I want to vectorize this so that I can do the mass fraction interpolation
    # in a vectorized fashion.
    array = np.array(value, ndmin=1)
    with np.errstate(divide='ignore'):
        log_vals = np.log10(array)
    log_vals[np.isneginf(log_vals)] = -6
    return log_vals
    # chosen to that is is below the lowest metallicity value
    # in any of the models (lowest is 10^-4 z_sun in ww95).

# store the known metallicity values
z_values_nomoto = [0, 0.001, 0.004, 0.02]
# to interpolate we need the log of that
log_z_nomoto = _metallicity_log(z_values_nomoto)

# we know the metallicity of the models WW95 used
z_sun = 0.02
z_values_ww = [0, (10**-4) * z_sun, 0.01 * z_sun, 0.1 * z_sun, z_sun]
# to interpolate we need the log of that
log_z_ww = _metallicity_log(z_values_ww)

def _interpolation_wrapper(metallicities, abundances):
    """
    Wraps the interpolation process, which is the same for all model creations.
    
    We interpolate in log of metallicity space. 
    
    :param metallicities: list of metallicities (not log of metallicities) that
                          corresponds with the abundances
    :param abundances: list of abundances for a given element. Each item 
                       corresponds with the metallicity in that location
    :return: scipy.interpolate.interp1D object that can be called to find the 
             appropriate abundance at the desired log(metallicity) value. 
             When doing things with this, be sure to convert to log of
             metallicity before calling this interpolation object. 
    """
    # also need to fix the extrapolation, so that it retuns
    # the values of the nearest model if the metallicity is
    # outside the range of the models themselves
    fill_values = (abundances[0], abundances[-1])
    # ^ assumes the abundances are in increasing metallicity
    log_met = _metallicity_log(metallicities)

    return interpolate.interp1d(log_met, abundances, kind="linear",
                                bounds_error=False, fill_value=fill_values)


class Yields(object):
    """Class containing yields from supernovae"""
    def __init__(self, model_set):
        """ Initialize the object, given the reference for the yields you'd like
        to use."""

        # the main functionality will be in two dictionaries. The
        # _abundances_interp one will hold interpolation objects, that are able
        # to get the abundance at any metallicity. Then once the metallicity is
        # set, we store the abundances at that metallicity in a second dict
        self.abundances = dict()
        self._abundances_interp = dict()

        # store the model set the user is using
        self.model_set = model_set

        # also store the default point of metallicity (this will actally be
        # filled in by the models when they are used
        self.metallicity_points = []

        # then we can initialize the model set they are using.
        if model_set == "test":
            self.make_test()
        elif model_set.startswith("iwamoto_99_Ia_"):
            self.make_iwamoto_99_Ia(_parse_iwamoto_model(model_set))
        elif model_set == "nomoto_06_II":
            self.make_nomoto_06_II()
        elif "imf" in model_set:
            if "nomoto" in model_set:
                if "ave" in model_set:
                    self.make_imf_integrated(my_nomoto_ave_file)
                elif "no_hn" in model_set:
                    self.make_imf_integrated(my_nomoto_reg_file)
                elif "hn" in model_set:
                    self.make_imf_integrated(my_nomoto_hn_file)
                else:
                    raise ValueError("Not recognized.")
            elif "ww_95" in model_set:
                self.make_imf_integrated(my_ww_file)
            else:
                raise ValueError("Not recognized.")


        elif model_set.startswith("nomoto_06_II"):
            if "hn" in model_set:
                mass = model_set[-5:-3]
                self.make_individual_nomoto_hn(mass)
            else:
                mass = model_set[-2:]
                self.make_individual_nomoto_regular(mass)

        elif model_set.startswith("ww_95_II"):
            model = model_set[9:]
            self.make_individal_ww95(model)
        else:
            raise ValueError("This model is not supported. Make sure you\n" +
                             "entered it correctly.")

        # and that the user so far has not specified a normalization
        self.has_normalization = False

        # fix the WW 95 iron thing
        if "ww_95" in model_set:
            self._handle_iron_ww()

        # all model sets have a zero metallicity option, so set the initial
        # metallicity to that. This takes care of the _set_member() call too.
        self.set_metallicity(0, initial=True)

        # we then want to keep track of the initial total metals
        self.total_metals = self.metals_sum()

        # then create the mass fraction objects
        self._create_mass_fractions()

    def set_metallicity(self, metallicity, initial=False):
        """Sets the metallicity (Z). This is needed since the models depend on Z
        
        :param metallicity: The metallicity (Z) at which to calculate the 
                            supernova yields. 
        :param initial: Whether or not this is the first time this is done. 
                        Since this is called in the __init__() function, as a 
                        user this will always be False, which is the default 
                        value. Do not set True here.
        """
        # first do error checking
        if not 0 <= metallicity <= 1:
            raise ValueError("Metallicity must be between zero and one.")

        # go through all values, and call them at the metallicity requested,
        # then put those values into the abundances dictionary
        for isotope in self._abundances_interp:
            # we interpolate in log of metallicity space, so we need to
            # take the log and use it in the interpolation
            met_log = _metallicity_log(metallicity)

            new_value = self._abundances_interp[isotope](met_log)
            self.abundances[isotope] = np.asscalar(new_value)

        # we then need to normalize the the old total abundance if we are doing
        # this any time other than the very first time we set the metallicity,
        # since then there will be no total_metals already existing. If this is
        # the first time, we have to take care of the _set_members(), which
        # something the normalize_metals function would do for us.
        self._set_members()
        if self.has_normalization:
            self.normalize_metals(self.total_metals)


        # also store the metallicity
        self.metallicity = metallicity

    def _set_members(self):
        """Puts the elements of the dictionary as attributes of the object

        This must be done after every time we change things"""

        self._sum_elements()
        for key in self.abundances:
            setattr(self, key, self.abundances[key])

    def _sum_elements(self):
        """Creates the sum of each element over all isotopes"""
        for isotope in self.abundances.copy():  # have to copy so the original
                                                # doesn't change size
            if "_" not in isotope:  # we have a summed element already
                continue
            element, mass_num = isotope.split("_")

            # add it
            this_element = [value for key, value in self.abundances.items()
                            if element + "_" in key]
            # element + "_" is needed to that "H" doesn't match with "He".
            self.abundances[element] = np.sum(this_element)

    def metals_sum(self):
        """Gets the sum of the metals in the yields. This includes everything 
        other than H and He."""
        total_metals = 0
        for isotope in self.abundances:
            if "_" not in isotope and isotope not in ["H", "He"]:
            # use the sums we already created to make this easier.
                total_metals += self.abundances[isotope]

        return float(total_metals)

    def normalize_metals(self, total_metals):
        """Takes the yields and normalizes them to have some total metal output.
        :param total_metals: total metal abundance, in solar masses.
        :type total_metals: float
        """
        # first get the original sum of metals, so we know
        total_before = self.metals_sum()
        scale_factor = total_metals / total_before
        for key in self.abundances:
            self.abundances[key] *= scale_factor

        self._set_members()

        # we then want to keep track of this going forward
        self.total_metals = total_metals

        self.has_normalization = True

    def _create_mass_fractions(self):
        """Creates a dictionary of mass fraction interpolation objects that
        can be called at any metallicity. 
        
        The values here are the fraction of metals that a given isotope makes
        up. It's mass(isotope) / total_metals."""

        # store the initial metallicity, so we can restore that when we're done
        initial_z = self.metallicity

        # first create the dictionary. We have one that is under the hood, and
        # holds the objects where the interpolation is done in log Z space
        self._mass_fractions_log_z = dict()

        # need the log of the model's z vals to interpolate with later
        log_z_vals = _metallicity_log(self.metallicity_points)

        temp_mass_frac_storage = defaultdict(list)
        for z in self.metallicity_points:
            self.set_metallicity(z)
            tot_metals = self.metals_sum()
            # need to do this for all elements
            for isotope in self.abundances:
                # to calculate the mass fraction, we divide the mass of this
                # isotope by the total mass in metals
                this_frac = self.abundances[isotope] / tot_metals
                temp_mass_frac_storage[isotope].append(this_frac)


        # then create the interpolation object
        for isotope, mass_fractions in temp_mass_frac_storage.items():
            interp_obj = _interpolation_wrapper(self.metallicity_points,
                                                mass_fractions)
            self._mass_fractions_log_z[isotope] = interp_obj

        # then restore the metallicity
        self.set_metallicity(initial_z)

    def mass_fraction(self, isotope, metallicity):
        """Get the mass fraction for a particular isotope. """

        # this needs to be done because the function that does the interpolation
        # inteprolates in log(Z) space, but the user won't want to mess with
        # that, so we have to transform the metallicity before calling it.
        # That's all we do here.
        log_z = _metallicity_log(metallicity)
        return self._mass_fractions_log_z[isotope](log_z)

    def make_test(self):
        # totally arbitrary values for testing
        self.metallicity_points = [0, 1]
        metallicities = _metallicity_log(self.metallicity_points)
        self._abundances_interp["H_1"] = interpolate.interp1d(metallicities,
                                                              [1, 2])
        self._abundances_interp["He_2"] = interpolate.interp1d(metallicities,
                                                              [2, 3])
        self._abundances_interp["Li_3"] = interpolate.interp1d(metallicities,
                                                              [3, 4])
        self._abundances_interp["Be_4"] = interpolate.interp1d(metallicities,
                                                              [4, 5])
        self._abundances_interp["B_5"] = interpolate.interp1d(metallicities,
                                                              [5, 6])
        self._abundances_interp["C_6"] = interpolate.interp1d(metallicities,
                                                              [6, 7])
        self._abundances_interp["N_7"] = interpolate.interp1d(metallicities,
                                                              [7, 8])
        self._abundances_interp["O_8"] = interpolate.interp1d(metallicities,
                                                              [8, 9])
        self._abundances_interp["F_9"] = interpolate.interp1d(metallicities,
                                                              [9, 10])
        self._abundances_interp["Na_10"] = interpolate.interp1d(metallicities,
                                                              [10, 11])

    def make_iwamoto_99_Ia(self, model="W7"):
        """Populates the object with the type Ia supernova abundances from
        Iwamoto et al 1999

        :param model: which model from the paper to use. The options are "W7", 
                      "W70", "WDD1", "WDD2", "WDD3", "CDD1", "CDD2". The "W7" 
                      model is typically the one that is used the most.
        """
        self.metallicity_points = [0, 1]
        # get the index of the correct column
        column_idxs = {"W7":2, "W70":3, "WDD1":4, "WDD2":5, "WDD3":6, 
                       "CDD1":7, "CDD2":8}
        our_idx = column_idxs[model]

        # then iterate through each line and handle it appropriately
        with open(_get_data_path(iwamoto_file), "r") as in_file:
            for line in in_file:
                # ignore the comments
                if not line.startswith("#"):
                    # We then need to get the appropriate values from the line.
                    # to do this we split it on spaces, then use the index
                    # we had above
                    split_line = line.split()
                    element = split_line[0]
                    abundance = split_line[our_idx]

                    # the elements are formatted in LaTeX in the table, so we
                    # need to format it properly
                    formatted_element = _parse_iwamoto_element(element)
                    # We then need to make the interpolation object. Since this
                    # will be the same at all metallicities, this is easy
                    interp_obj = _interpolation_wrapper(self.metallicity_points,
                                                        [float(abundance)]*2)

                    self._abundances_interp[formatted_element] = interp_obj

    def make_nomoto_06_II(self):
        """Populates the model with the yields from the Nomoto 2006 models"""
        self.metallicity_points = z_values_nomoto

        # then iterate through each line and handle it appropriately
        with open(_get_data_path(nomoto_file), "r") as in_file:
            for line in in_file:
                # ignore the comments
                if not line.startswith("#"):
                    # We then need to get the appropriate values from the line.
                    # to do this we split it on spaces, then we know where
                    # everything is
                    split_line = line.split()
                    mass_number = split_line[0]
                    atomic_name = split_line[1]
                    these_abundances = split_line[2:]

                    # We can then parse the string to get the elemental format
                    # we need
                    formatted_element = _parse_nomoto_element(mass_number,
                                                              atomic_name)

                    interp_obj = _interpolation_wrapper(self.metallicity_points,
                                                        these_abundances)
                    self._abundances_interp[formatted_element] = interp_obj

    def make_individual_nomoto_regular(self, mass):
        """Populates the model with the yields from the Nomoto 2006
        individual supernova values, not the IMF integrated ones."""
        self.metallicity_points = z_values_nomoto
        #we then need to open a bunch of files with all this data
        z_0_0_file = open(_get_data_path(nomoto_ind_0), "r")
        z_0_001_file = open(_get_data_path(nomoto_ind_0_001), "r")
        z_0_004_file = open(_get_data_path(nomoto_ind_0_004), "r")
        z_0_02_file = open(_get_data_path(nomoto_ind_0_02), "r")

        # we know the format of the file, so we know which column the mass we
        # want is in. We store those indexes here
        idxs = {"13": 1, "15": 2, "18": 3, "20": 4, "25": 5, "30": 6, "40": 7}
        #then get the one we want
        try:
            idx = idxs[mass]
        except KeyError:
            raise ValueError("This model was not found:"
                             " nomoto_II_{}".format(mass))

        # then we can iterate through the files. Each file is a given
        # metallicity and has all mass models, we we want to iterate through
        # all files at the same time
        for z_0_0, z_0_001, z_0_004, z_0_002 in zip(z_0_0_file.readlines(),
                                                    z_0_001_file.readlines(),
                                                    z_0_004_file.readlines(),
                                                    z_0_02_file.readlines()):
            # only get the rows that matter
            if z_0_0.split()[0] in ["M", "E", "Mcut"]:
                continue

            # get the right column from each file
            elt = z_0_0.split()[0]
            items = [row.split()[idx] for row in [z_0_0, z_0_001,
                                                  z_0_004, z_0_002]]

            #parse the element name
            elt = _parse_nomoto_individual_element(elt)

            # then get the interpolation object
            interp_obj = _interpolation_wrapper(self.metallicity_points, items)
            self._abundances_interp[elt] = interp_obj

    def make_individual_nomoto_hn(self, mass):
        """Populates the model with the yields from the Nomoto 2006
        individual supernova values, not the IMF integrated ones."""
        self.metallicity_points = z_values_nomoto

        #we then need to open a bunch of files with all this data
        z_0_0_file = open(_get_data_path(nomoto_ind_0_hn), "r")
        z_0_001_file = open(_get_data_path(nomoto_ind_0_001_hn), "r")
        z_0_004_file = open(_get_data_path(nomoto_ind_0_004_hn), "r")
        z_0_02_file = open(_get_data_path(nomoto_ind_0_02_hn), "r")

        # we know the format of the file, so we know which column the mass we
        # want is in. We store those indexes here
        idxs = {"20": 1, "25": 2, "30": 3, "40": 4}
        #then get the one we want
        try:
            idx = idxs[mass]
        except KeyError:
            raise ValueError("This model was not found:"
                             " nomoto_II_{}_hn".format(mass))

        # then we can iterate through the files. Each file is a given
        # metallicity and has all mass models, we we want to iterate through
        # all files at the same time
        for z_0_0, z_0_001, z_0_004, z_0_002 in zip(z_0_0_file, z_0_001_file,
                                                    z_0_004_file, z_0_02_file):
            # only get the rows that matter
            if z_0_0.split()[0] in ["M", "E", "Mcut"]:
                continue

            # get the right column from each file
            elt = z_0_0.split()[0]
            items = [row.split()[idx] for row in [z_0_0, z_0_001,
                                                  z_0_004, z_0_002]]

            #parse the element name
            elt = _parse_nomoto_individual_element(elt)

            # then get the interpolation object
            interp_obj = _interpolation_wrapper(self.metallicity_points,
                                                items)
            self._abundances_interp[elt] = interp_obj

    def make_individal_ww95(self, model):
        """Populates the model with the data from the individual 
        WW95 models"""
        self.metallicity_points = z_values_ww

        # we then need to open a bunch of files with all this data. Depending
        # on the mass of the model, we will have to open different files
        if int(model[0:2]) < 30:
            z_0_file = open(_get_data_path(ww_ind_0_a), "r")
            z_1_file = open(_get_data_path(ww_ind_4_sol_a), "r")
            z_2_file = open(_get_data_path(ww_ind_0_01_sol_a), "r")
            z_3_file = open(_get_data_path(ww_ind_0_1_sol_a), "r")
            z_4_file = open(_get_data_path(ww_ind_sol_a), "r")
        else:
            z_0_file = open(_get_data_path(ww_ind_0_b), "r")
            z_1_file = open(_get_data_path(ww_ind_4_sol_b), "r")
            z_2_file = open(_get_data_path(ww_ind_0_01_sol_b), "r")
            z_3_file = open(_get_data_path(ww_ind_0_1_sol_b), "r")
            z_4_file = open(_get_data_path(ww_ind_sol_b), "r")

        # we know the format of the file, so we know which column the mass we
        # want is in. We store those indexes here
        idxs_0 = {"12A": 1, "13A": 2, "15A": 3, "18A": 4, "20A": 5, "22A": 6,
                  "25A": 7, "25B": 8}
        idxs_1 = {"12A": 1, "13A": 2, "15A": 3, "18A": 4, "20A": 5, "22A": 6,
                  "25A": 7}
        idxs_4 = {"11A": 1, "12A": 2, "13A": 3, "15A": 4, "18A": 5, "19A": 6,
                  "20A": 7, "22A": 8, "25A": 9}
        all_high_mass = {"30A": 1, "30B": 2, "35A": 3, "35B": 4, "35C": 5,
                         "40A": 6, "40B": 7, "40C": 8}
        idxs_0.update(all_high_mass)
        idxs_1.update(all_high_mass)
        idxs_4.update(all_high_mass)

        # there are certain models that need special handling, since they are
        # only present in some metallicity sets
        if "11A" in model:  # only present in solar
            self._handle_different_ww95(z_4_file, idxs_4[model])
            return
        elif "19A" in model:  # only present in solar
            self._handle_different_ww95(z_4_file, idxs_4[model])
            return
        elif "25B" in model:  # only present in zero metallicity
            self._handle_different_ww95(z_0_file, idxs_0[model])
            return

        # then get the one we want
        try:
            idx_0 = idxs_0[model]
            idx_1 = idxs_1[model]
            idx_2 = idx_1  # 2 and 3 are the same as 1
            idx_3 = idx_1
            idx_4 = idxs_4[model]
        except KeyError:
            raise ValueError("This is not a valid WW95 model. ")

        # We will do each file one at a time and store the results as we go
        # before doing the interpolation later. We go in order of increasing
        # metallicity, so that this is the same as the metallicity list we made
        # earlier. I have since redone the files and this problem should be
        # fixed, but the code still works, so oh well.
        elements = defaultdict(list)
        for idx, this_file in zip([idx_0, idx_1, idx_2, idx_3, idx_4],
                                  [z_0_file, z_1_file, z_2_file, z_3_file,
                                   z_4_file]):
            for row in this_file:
                # only get the rows that matter
                if row.split()[0] in ["elt", "KE", "Mass"]:
                    continue

                # get the proper columns from the file
                elt = row.split()[0]
                item = row.split()[idx]
                # parse the item. The format here is the same as Nomoto
                elt = _parse_nomoto_individual_element(elt)

                # then put this in the dictionary
                elements[elt].append(item)

        # we can then create the interpolation objects and assign them to the
        # dictionary for the object
        for elt, item in elements.items():
            # We then need to make the interpolation object.
            interp_obj = _interpolation_wrapper(self.metallicity_points, item)

            self._abundances_interp[elt] = interp_obj

    def _handle_different_ww95(self, in_file, idx):

        # then iterate through each line and handle it appropriately
        elements = dict()
        for row in in_file:
            # only get the rows that matter
            if row.split()[0] in ["elt", "KE", "Mass"]:
                continue

            # get the proper columns from the file
            elt = row.split()[0]
            item = row.split()[idx]
            # parse the item. The format here is the same as Nomoto
            elt = _parse_nomoto_individual_element(elt)

            # then put this in the dictionary
            elements[elt] = item

        # we can then create the interpolation objects and assign them to the
        # dictionary for the object
        for elt, item in elements.items():
            # We then need to make the interpolation object.

            try:
                interp_obj = _interpolation_wrapper([0, 1], [item] * 2)
            except ValueError:
                print([0, 1], [item] * 2)
                raise KeyError
            self._abundances_interp[elt] = interp_obj

    def make_imf_integrated(self, filename):
        # we need to get the metallicities used here
        if "nomoto" in filename:
            self.metallicity_points = z_values_nomoto
        else:  # use WW
            self.metallicity_points = z_values_ww

        # then iterate through each line and handle it appropriately
        with open(_get_data_path(filename), "r") as in_file:
            for line in in_file:
                # ignore the comments
                if not line.startswith("#"):
                    # We then need to get the appropriate values from the line.
                    # to do this we split it on spaces, then we know where
                    # everything is
                    split_line = line.split()
                    elt = split_line[0]
                    these_abundances = split_line[1:]

                    # the element is already formatted properly, so we don't
                    # have to change anything there

                    # We then need to make the interpolation object.
                    interp_obj = _interpolation_wrapper(self.metallicity_points,
                                                        these_abundances)
                    self._abundances_interp[elt] = interp_obj

    def _handle_iron_ww(self):
        """In the WW 95 yields, the 56 Ni should decay to Fe 56 after a longer
        period of time, but the evolution stops too early. To fix this, we add
        all the 56Ni to the 56Fe. The Iron is also too high, so we divide
        it by two."""
        real_56_fe_abundances = []
        real_56_ni_abundances = []
        for z in self.metallicity_points:
            self.set_metallicity(z)
            real_56_fe_abundances.append((self.Ni_56 + self.Fe_56) / 2.0)
            real_56_ni_abundances.append(0)

        ni_56_interp = _interpolation_wrapper(self.metallicity_points,
                                              real_56_ni_abundances)
        fe_56_interp = _interpolation_wrapper(self.metallicity_points,
                                              real_56_fe_abundances)

        self._abundances_interp["Ni_56"] = ni_56_interp
        self._abundances_interp["Fe_56"] = fe_56_interp