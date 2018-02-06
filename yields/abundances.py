import os

import numpy as np
from astropy import table

import yields


# need to get the solar abundances
def create_solar_metal_fractions():
    this_dir = os.path.dirname(__file__)
    solar_file = this_dir + '/data/solar_abundance.txt'
    solar = table.Table.read(solar_file, format="ascii",
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

    @classmethod
    def hydrogen(cls, Z_tot):
        """
        Calculate the fraction of mass in H for a given Z. This assumes a solar
        ratio of Helium to Hydrogen, and we use the following math:

        X + Y + Z = 1
        X (1 + Y/X) = 1 - Z
        X = (1 - Z)/(1 + Y/X)

        :param Z_tot: Total metallicity.
        :return: Mass fraction of H.
        """
        Y = cls.solar_metal_fractions["He"]
        X = cls.solar_metal_fractions["H"]

        return (1 - Z_tot) / (1 + Y/X)

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
        """Error checking on the user metallicity value."""
        # turn to array if not already.
        if not isinstance(Z_Ia, np.ndarray):
            try:
                len(Z_Ia) # checks for scalars
            except TypeError:
                Z_Ia = np.array([Z_Ia])  # scalars need to be in a list
            else:
                Z_Ia = np.array(Z_Ia)  # lists don't.
        # same thing for Type II
        if not isinstance(Z_II, np.ndarray):
            try:
                len(Z_II)
            except TypeError:
                Z_II = np.array([Z_II])
            else:
                Z_II = np.array(Z_II)

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
        """Return a float if we have a one element array, otherwise return
        the whole array. """
        if len(array) == 1:
            return float(array[0])
        else:
            return array

    def z_on_h(self, Z_Ia, Z_II):
        """Calculate [Z/H].

        .. math::
            [Z/H] = \log_{10} \left[ \frac{Z_{tot \star}}{1 - Z_{tot \star}} /
            \frac{Z_\odot}{1 - Z_\odot} \right]

        This is basically the sum of metals divided by sum of not metals for
        both the sun and the star. Not metals is a proxy for Hydrogen, since
        we assume cosmic abundances for both (not quite right, but not too bad).

        :param Z_Ia: metallicity from type Ia supernovae
        :param Z_II: metallicity from type II supernovae
        :returns: [Z/H]
        :rtype: float if a single metallicity is passed, otherwise np.ndarray
        """
        Z_Ia, Z_II = self._err_checking_z(Z_Ia, Z_II)

        Z_tot = Z_Ia + Z_II
        try:
            star_frac = Z_tot / self.hydrogen(Z_tot)
            sun_frac  = self.Z_sun / self.hydrogen(self.Z_sun)
        except ZeroDivisionError:
            return np.inf

        return self._rtype(np.log10(star_frac / sun_frac))

    def x_on_h(self, element, Z_Ia, Z_II):
        """Calculate [X/H].

        This is calculated in the following way.

        .. math::
            [X/H] = \log_{10} \left[ \frac{Z_\star^{Ia}
            f_X^{Ia} + Z_\star^{II} f_X^{II}}{
            1 - Z_{tot \star}} / \frac{Z_\odot f_{X \odot}}{1 - Z_\odot} \right]


        Where f is the fraction of the total metals element x takes up for
        either the type Ia or II yields.

        This calculation is basically the sum of the mass in that metal divided
        by the mass of not metals for both the sun and the star. This works
        because we assume a cosmic abundance for H, making the mass that isn't
        in metals a proxy for H.

        :param element: Element to be used in place of X.
        :type element: str
        :param Z_Ia: metallicity from type Ia supernovae
        :param Z_II: metallicity from type II supernovae
        :returns: [X/H]
        :rtype: float if a single metallicity is passed, otherwise np.ndarray
        """
        Z_Ia, Z_II = self._err_checking_z(Z_Ia, Z_II)

        # get the metal mass fractions
        f_Ia = self.yields_Ia.mass_fraction(element, Z_Ia)
        f_II = self.yields_II.mass_fraction(element, Z_II)

        try:
            star_num = Z_Ia * f_Ia + Z_II * f_II
            star_denom = self.hydrogen(Z_Ia + Z_II)
            star_frac = star_num / star_denom
        except ZeroDivisionError:
            return np.inf

        sun_num = self.Z_sun * self.solar_metal_fractions[element]
        sun_denom = self.hydrogen(self.Z_sun)
        sun_frac = sun_num / sun_denom

        return self._rtype(np.log10(star_frac / sun_frac))

    def x_on_fe(self, element, Z_Ia, Z_II):
        """Calculate [X/Fe].

        This is calculated in the following way.

        .. math::
            [X/Fe] = \log_{10} \left[ \frac{
            Z_\star^{Ia}f_X^{Ia} + Z_\star^{II} f_X^{II}}{
            Z_\star^{Ia}f_{Fe}^{Ia} + Z_\star^{II} f_{Fe}^{II}} /
            \frac{f_{X \odot}}{f_{Fe \odot}} \right]

        Where f is the fraction of the total metals element x takes up for
        either the type Ia or II yields.

        This calculation is basically the sum of the mass in that metal divided
        by the mass of iron for both the sun and the star.

        :param element: Element to be used in place of X.
        :type element: str
        :param Z_Ia: metallicity from type Ia supernovae
        :param Z_II: metallicity from type II supernovae
        :returns: [X/Fe]
        :rtype: float if a single metallicity is passed, otherwise np.ndarray
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

        :param Z_Ia: metallicity from type Ia supernovae
        :param Z_II: metallicity from type II supernovae
        :returns: value of log(Z/Z_sun)
        :rtype: float if a single metallicity is passed, otherwise np.ndarray
        """
        Z_Ia, Z_II = self._err_checking_z(Z_Ia, Z_II)

        Z_tot = Z_Ia + Z_II
        return self._rtype(np.log10(Z_tot / self.Z_sun))
