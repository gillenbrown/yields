import os

import numpy as np

import yields


# need to get the solar abundances
def create_solar_metal_fractions():
    this_dir = os.path.dirname(__file__)
    solar_file = this_dir + '/data/solar_abundance.txt'
    solar = np.genfromtxt(solar_file, dtype=None,
                          names=['Natom', 'name', 'fN', 'log', 'f_mass'])
    z_mass = np.sum(solar["f_mass"][2:])

    metal_fractions = dict()
    for row in solar:
        elt = row["name"]
        f_mass = row["f_mass"]
        metal_fractions[elt] = f_mass / z_mass

    return z_mass, metal_fractions


class Abundances(object):
    """Holds infomation about the abundances of an object. """
    # get some of the solar information
    Z_sun, solar_metal_fractions = create_solar_metal_fractions()

    def __init__(self, II_type="nomoto"):
        """Create an abundance object.

        :param II_type: Which model of Type Ia supernovae to use. Either
                        "nomtoto" or "ww".
        :type II_type: str
        :returns: None, but sets attributes.
        """

        # create the yield objects that will be used to calculate the SN yields
        self.yields_Ia = yields.Yields("iwamoto_99_Ia_W7")
        if II_type == "nomoto":
            self.yields_II = yields.Yields("nomoto_06_II_imf_ave")
        elif II_type == "ww":
            self.yields_II = yields.Yields("ww_95_imf_ave")

    def _err_checking_z(self, Z_Ia, Z_II):
        if not isinstance(Z_Ia, np.ndarray):
            try:
                len(Z_Ia)
            except TypeError:
                Z_Ia = np.array([Z_Ia])
            else:
                Z_Ia = np.array(Z_Ia)
        if not isinstance(Z_II, np.ndarray):
            try:
                len(Z_II)
            except TypeError:
                Z_II = np.array([Z_II])
            else:
                Z_II = np.array(Z_II)

        # do error checking.
        # all arrays must be the same length
        if not len(Z_Ia) == len(Z_II):
            raise ValueError("All arrays must be the same length. ")
        # the metallicity must be between 0 and 1.
        for z_type in [Z_Ia, Z_II]:
            if any(z_type < 0) or any(z_type > 1):
                raise ValueError("Metallicity must be between 0 and 1.")

        Z_tot = Z_Ia + Z_II
        # also have to check that the total metallicity isn't larger than one.
        if any(Z_tot < 0) or any(Z_tot > 1):
            raise ValueError("Total metallicity can't be larger than one. ")

        return Z_Ia, Z_II

    def _rtype(self, array):
        if len(array) == 1:
            return float(array[0])
        else:
            return array

    def z_on_h(self, Z_Ia, Z_II):
        """Calculate the Z on H value for this collection of stars, by
        dividing the total Z by the total H.

        This is calculated in the following way. The derivation for this is
        in my notebook, but here is the important equation.

        .. math::
            [Z/H] = \log_{10} \left[ \frac{\sum_\star M_\star Z_{tot \star}}
            {\sum_\star M_\star (1 - Z_{tot \star})}
            \frac{1 - Z_\odot}{Z_\odot} \right]

        This is basically the sum of metals divided by sum of not metals for
        both the sun and the stars. Not metals is a proxy for Hydrogen, since
        we assume cosmic abundances for both (not quite right, but not too bad).

        :returns: [Z/H] value for this collection of stars
        :rtype: float
        """
        Z_Ia, Z_II = self._err_checking_z(Z_Ia, Z_II)

        Z_tot = Z_Ia + Z_II
        try:
            star_frac = Z_tot / (1.0 - Z_tot)
            sun_frac  = self.Z_sun / (1.0 - self.Z_sun)
        except ZeroDivisionError:
            return np.inf

        return self._rtype(np.log10(star_frac / sun_frac))

    def x_on_h(self, element, Z_Ia, Z_II):
        """Calculate the [X/H] value for this collection of stars.

        This is calculated in the following way.

        .. math::
            [X/H] = \log_{10} \left[ \frac{\sum_\star M_\star (Z_\star^{Ia}
            f_X^{Ia} + Z_\star^{II} f_X^{II})}{\sum_\star M_\star
            (1 - Z_{tot \star})}\frac{1 - Z_\odot}{Z_\odot f_{X \odot}} \right]

        Where f is the fraction of the total metals element x takes up for
        either the type Ia or II yields.

        This calculation is basically the sum of the mass in that metal divided
        by the mass of not metals for both the sun and the star. This works
        because we assume a cosmic abundance for H, making the mass that isn't
        in metals a proxy for H.

        :param element: Element to be used in place of X.
        :type element: str
        :returns: Value of [X/H] for the given element.
        :rtype: float
        """
        Z_Ia, Z_II = self._err_checking_z(Z_Ia, Z_II)

        # get the metal mass fractions
        f_Ia = self.yields_Ia.mass_fraction(element, Z_Ia)
        f_II = self.yields_II.mass_fraction(element, Z_II)

        try:
            star_num = Z_Ia * f_Ia + Z_II * f_II
            star_denom = 1.0 - (Z_Ia + Z_II)
            star_frac = star_num / star_denom
        except ZeroDivisionError:
            return np.inf

        sun_num = self.Z_sun * self.solar_metal_fractions[element]
        sun_denom = 1.0 - self.Z_sun
        sun_frac = sun_num / sun_denom

        return self._rtype(np.log10(star_frac / sun_frac))

    def x_on_fe(self, element, Z_Ia, Z_II):
        """Calculate the [X/Fe] value for this collection of stars.

        This is calculated in the following way.

        .. math::
            [X/Fe] = \log_{10} \left[ \frac{\sum_\star M_\star
            (Z_\star^{Ia}f_X^{Ia} + Z_\star^{II} f_X^{II})}{\sum_\star M_\star
            (Z_\star^{Ia}f_{Fe}^{Ia} + Z_\star^{II} f_{Fe}^{II})}
            \frac{f_{Fe \odot}}{f_{X \odot}} \right]

        Where f is the fraction of the total metals element x takes up for
        either the type Ia or II yields.

        This calculation is basically the sum of the mass in that metal divided
        by the mass of iron for both the sun and the star.

        :param element: Element to be used in place of X.
        :type element: str
        :returns: Value of [X/Fe] for the given element.
        :rtype: float
        """
        Z_Ia, Z_II = self._err_checking_z(Z_Ia, Z_II)

        # get the metal mass fractions
        f_Ia_x = self.yields_Ia.mass_fraction(element, Z_Ia)
        f_II_x = self.yields_II.mass_fraction(element, Z_II)
        f_Ia_Fe = self.yields_Ia.mass_fraction("Fe", Z_Ia)
        f_II_Fe = self.yields_II.mass_fraction("Fe", Z_II)

        try:
            star_num = Z_Ia * f_Ia_x + Z_II * f_II_x
            star_denom = Z_Ia * f_Ia_Fe + Z_II * f_II_Fe
            star_frac = star_num / star_denom
        except ZeroDivisionError:
            return np.inf

        sun_num = self.solar_metal_fractions[element]
        sun_denom = self.solar_metal_fractions["Fe"]
        sun_frac = sun_num / sun_denom

        return self._rtype(np.log10(star_frac / sun_frac))

    def log_z_over_z_sun(self, Z_Ia, Z_II):
        """Returns the value of log(Z/Z_sun).

        This is a pretty straightforward calculation. We just take the total
        mass in metals and divide by the total stellar mass to get the
        overall metallicity of the star particles, then divide that by the
        solar metallicity.

        :returns: value of log(Z/Z_sun)
        :rtype: float
        """
        Z_Ia, Z_II = self._err_checking_z(Z_Ia, Z_II)

        Z_tot = Z_Ia + Z_II
        return self._rtype(np.log10(Z_tot / self.Z_sun))
