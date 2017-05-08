import os

from scipy import interpolate
import numpy as np

iwamoto_file = "iwamoto_99_Ia_yields.txt"
nomoto_file = "nomoto_06_imf_weighted_II.txt"


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
    if value == 0:
        return -4  # obtained from Figure 11 in Nomoto et al 2006
    else:
        return np.log10(value)


class Yields(object):
    """Class containing yields from supernovae"""
    def __init__(self, model_set):
        """ Initialize the object, given the reference for the yields you'd like
        to use."""

        # the main functionality will be in two dictionaries. The
        # _abundances_interp one will hold interpolation objects, that are able
        # to get the abundance at any metallicity. Then once the metallicity is
        # set, we store the abundances at that metallicity in a second dict
        self._abundances = dict()
        self._abundances_interp = dict()

        # store the model set the user is using
        self.model_set = model_set

        # then we can initialize the model set they are using.
        if model_set == "test":
            self.make_test()
        elif model_set.startswith("iwamoto_99_Ia_"):
            self.make_iwamoto_99_Ia(_parse_iwamoto_model(model_set))
        elif model_set == "nomoto_06_II":
            self.make_nomoto_06_II()

        # all model sets have a zero metallicity option, so set the initial
        # metallicity to that. This takes care of the _set_member() call too.
        self.set_metallicity(0, initial=True)

        # we then want to keep track of the initial total metals
        self.total_metals = self._metals_sum()

        # and that the user so far has not specified a normalization
        self.has_normalization = False

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
            self._abundances[isotope] = np.asscalar(new_value)

        # we then need to normalize the the old total abundance if we are doing
        # this any time other than the very first time we set the metallicity,
        # since then there will be no total_metals already existing. If this is
        # the first time, we have to take care of the _set_members(), which
        # something the normalize_metals function would do for us.
        if initial or not self.has_normalization:
            self._set_members()
        elif self.has_normalization:
            self.normalize_metals(self.total_metals)
        # also store the metallicity
        self.metallicity = metallicity

    def _set_members(self):
        """Puts the elements of the dictionary as attributes of the object

        This must be done after every time we change things"""

        for key in self._abundances:
            setattr(self, key, self._abundances[key])

    def _metals_sum(self):
        """Gets the sum of the metals in the yields. This includes everything 
        other than H and He."""
        total_metals = 0
        for isotope in self._abundances:
            if "H_" not in isotope and "He_" not in isotope:
                total_metals += self._abundances[isotope]

        return float(total_metals)

    def normalize_metals(self, total_metals):
        """Takes the yields and normalizes them to have some total metal output.
        :param total_metals: total metal abundance, in solar masses.
        :type total_metals: float
        """
        # first get the original sum of metals, so we know 
        total_before = self._metals_sum()
        scale_factor = total_metals / total_before
        for key in self._abundances:
            self._abundances[key] *= scale_factor

        self._set_members()

        # we then want to keep track of this going forward
        self.total_metals = total_metals

        self.has_normalization = True

    def make_test(self):
        # totally arbitrary values for testing
        metallicities = [_metallicity_log(0), _metallicity_log(1)]
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
                    interp_obj = interpolate.interp1d([_metallicity_log(0),
                                                       _metallicity_log(1)],
                                                      [float(abundance)]*2,
                                                      kind="linear")
                    self._abundances_interp[formatted_element] = interp_obj

    def make_nomoto_06_II(self):
        """Populates the model with the yields from the Nomoto 2006 models"""

        # we know the metallicity of the models Nomoto used
        metallicity_values = [0, 0.001, 0.004, 0.02]
        # to interpolate we need the log of that
        log_met_values = [_metallicity_log(met) for met in metallicity_values]

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
                    # We then need to make the interpolation object. It takes
                    # the metallicities and the corresponding abundances. We
                    # also need to fix the extrapolation, so that it retuns
                    # the values of the nearest model if the metallicity is
                    # outside the range of the models themselves
                    fill_values = (these_abundances[0], these_abundances[-1])
                    # ^ assumes the abundances are in increasing metallicity
                    interp_obj = interpolate.interp1d(log_met_values,
                                                      these_abundances,
                                                      kind="linear",
                                                      bounds_error=False,
                                                      fill_value=fill_values)
                    self._abundances_interp[formatted_element] = interp_obj
