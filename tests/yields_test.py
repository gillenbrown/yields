import yields_base
import pytest
import numpy as np

# create simple object for testing
@pytest.fixture
def yields_test_case():
    return yields_base.Yields("test")

def test_metals_sum(yields_test_case):
    assert yields_test_case.ejecta_sum(metal_only=True) == 55 - 3  # sum(3..10)
    # this is because the test has elements 1 to 10, with an abundance of 1 to
    # 10, and we exclude H and He
    assert yields_test_case.ejecta_sum(metal_only=False) == 55  # sum(1..10)
    # this is because the test has elements 1 to 10, with an abundance of 1 to
    # 10, and we exclude H and He

def test_normalize_metals(yields_test_case):
    """Test whether or not the normalization is working properly"""
    yields_test_case.normalize_metals(1)
    assert np.isclose(yields_test_case.ejecta_sum(metal_only=True), 1)
    assert np.isclose(yields_test_case.H_1, 1.0 / 52.0)  # 1 / sum(3..10)
    assert np.isclose(yields_test_case.F_9, 9.0 / 52.0)
    assert np.isclose(yields_test_case.Na_10, 10.0 / 52.0)
    # the total amount must be larger than the amount of metals we normalized 
    # to, since H and He are included. 
    assert sum(yields_test_case.abundances.values()) > 1.0

    # normalize to a value other than 1
    yields_test_case.normalize_metals(25.0)
    assert np.isclose(yields_test_case.ejecta_sum(metal_only=True), 25.0)
    assert np.isclose(yields_test_case.H_1, 25.0 / 52.0) # 1 / sum(1..10)
    assert np.isclose(yields_test_case.F_9, 9.0 * 25.0 / 52.0)
    assert np.isclose(yields_test_case.Na_10, 10.0 * 25.0 / 52.0)
    # that one required isclose for whatever reason. 
    assert sum(yields_test_case.abundances.values()) > 25.0

def test_set_metallicity_error_checking(yields_test_case):
    """Metallicities are only vaild between zero and one."""
    with pytest.raises(ValueError):
        yields_test_case.set_metallicity(2)
    with pytest.raises(ValueError):
        yields_test_case.set_metallicity(-0.001)
    with pytest.raises(ValueError):
        yields_test_case.set_metallicity(1.001)

def test_interpolate_z_test(yields_test_case):
    """Test whether the interpolation is working correctly in the test case
    
    In the test case, each isotope goes between the atomic number and the 
    atomic number plus one at metallicities of 0 and 1. 
    So at Z=0, H=1, and at Z=1, H=2"""
    # at zero metallicity, each should be the atomic number
    yields_test_case.set_metallicity(0)
    assert yields_test_case.H_1 == 1.0
    assert yields_test_case.F_9 == 9.0
    assert yields_test_case.Na_10 == 10.0

    # then at all metallicity, each should be the atomic number plus one
    yields_test_case.set_metallicity(1.0)
    assert yields_test_case.H_1 == 2.0
    assert yields_test_case.F_9 == 10.0
    assert yields_test_case.Na_10 == 11.0

    # then test a point halfway in between in log space. This is a little weird,
    # since 0 is at -4 in log space according to my definition. So halfway
    # between -46and 0=log(1) in log space is log(x) = -3. The resulting
    # value should be atomic number + 0.5
    yields_test_case.set_metallicity(10**-3)
    assert yields_test_case.H_1 == 1.5
    assert yields_test_case.F_9 == 9.5
    assert yields_test_case.Na_10 == 10.5


test_trips = [["H", 1.0, 0.0], ["H", 2.0, 1.0],
              ["He", 2.0, 0.0], ["He", 3.0, 1.0],
              ["Li", 3.0, 0.0], ["Li", 4.0, 1.0],
              ["Na", 10.0, 0.0], ["Na", 11.0, 1.0]]
@pytest.mark.parametrize("elt,mass,z", test_trips)
def test_mass_fractions(yields_test_case, elt, mass, z):
    total_mass = {0: 55.0, 1.0: 65.0}
    real_mass_frac = mass / total_mass[z]
    test_mass_frac = yields_test_case.mass_fraction(elt, z, metal_only=False)
    assert real_mass_frac == test_mass_frac

@pytest.mark.parametrize("elt,mass,z", test_trips)
def test_metal_fractions(yields_test_case, elt, mass, z):
    total_mass = {0: 52.0, 1.0: 60.0}
    real_mass_frac = mass / total_mass[z]
    test_mass_frac = yields_test_case.mass_fraction(elt, z, metal_only=True)
    assert real_mass_frac == test_mass_frac


def test_get_iwamoto_path():
    """Tests the function that gets the path of the Iwamoto yields"""
    file_loc =  yields_base._get_data_path(yields_base.iwamoto_file)
    # I know the first line, so I can read that and see what it is
    iwamoto_file = open(file_loc, "r")
    assert iwamoto_file.readline() == "# Table 3 from Iwamoto et al 1999\n"

def test_get_nomoto_path():
    """Tests the function that gets the path of the Iwamoto yields"""
    file_loc =  yields_base._get_data_path(yields_base.nomoto_file)
    # I know the first line, so I can read that and see what it is
    iwamoto_file = open(file_loc, "r")
    assert iwamoto_file.readline() == "# Table 3 from Nomoto et al 2006\n"

def test_iwamoto_element_parsing():
    """Tests turning the format of the Iwamoto output into the format this
    class needs"""
    assert yields_base._parse_iwamoto_element("^{8}O") == "O_8"
    assert yields_base._parse_iwamoto_element("^{12}C") == "C_12"
    assert yields_base._parse_iwamoto_element("^{55}Mn") == "Mn_55"
    assert yields_base._parse_iwamoto_element("^{68}Zn") == "Zn_68"

def test_iwamoto_model_parsing():
    """Tests getting the model itself out of the iwamoto name"""
    assert yields_base._parse_iwamoto_model("iwamoto_99_Ia_W7") == "W7"
    assert yields_base._parse_iwamoto_model("iwamoto_99_Ia_W70") == "W70"
    assert yields_base._parse_iwamoto_model("iwamoto_99_Ia_WDD1") == "WDD1"
    assert yields_base._parse_iwamoto_model("iwamoto_99_Ia_WDD2") == "WDD2"
    assert yields_base._parse_iwamoto_model("iwamoto_99_Ia_WDD3") == "WDD3"
    assert yields_base._parse_iwamoto_model("iwamoto_99_Ia_CDD1") == "CDD1"
    assert yields_base._parse_iwamoto_model("iwamoto_99_Ia_CDD2") == "CDD2"
    with pytest.raises(ValueError):
        yields_base._parse_iwamoto_model("iwamsdfs")
    with pytest.raises(ValueError):
        yields_base._parse_iwamoto_model("iwamoto_99_Ia_wer")  #not a valid model

def test_make_iwamoto_w7():
    iwamoto_test = yields_base.Yields("iwamoto_99_Ia_W7")
    for z in [0, 0.5, 1]:
        iwamoto_test.set_metallicity(z)
        assert iwamoto_test.C_12 == 4.83E-02
        assert iwamoto_test.Cl_35 == 1.37E-04
        assert iwamoto_test.Zn_68 == 1.74E-08
        assert iwamoto_test.Zn_64 == 1.06E-05
        with pytest.raises(AttributeError):
            iwamoto_test.U_135
        with pytest.raises(AttributeError):
            iwamoto_test.H_1

def test_make_iwamoto_w70():
    iwamoto_test = yields_base.Yields("iwamoto_99_Ia_W70")
    for z in [0, 0.5, 1]:
        iwamoto_test.set_metallicity(z)
        assert iwamoto_test.C_12 == 5.08E-02
        assert iwamoto_test.Cl_35 == 1.06E-05
        assert iwamoto_test.Zn_68 == 1.13E-08
        assert iwamoto_test.Zn_64 == 7.01E-05
        with pytest.raises(AttributeError):
            iwamoto_test.U_135
        with pytest.raises(AttributeError):
            iwamoto_test.H_1

def test_make_iwamoto_wdd1():
    iwamoto_test = yields_base.Yields("iwamoto_99_Ia_WDD1")
    for z in [0, 0.5, 1]:
        iwamoto_test.set_metallicity(z)
        assert iwamoto_test.C_12 == 5.42E-03
        assert iwamoto_test.Cl_35 == 9.28E-05
        assert iwamoto_test.Zn_68 == 7.44E-08
        assert iwamoto_test.Zn_64 == 3.71E-06
        with pytest.raises(AttributeError):
            iwamoto_test.U_135
        with pytest.raises(AttributeError):
            iwamoto_test.H_1

def test_make_iwamoto_wdd2():
    iwamoto_test = yields_base.Yields("iwamoto_99_Ia_WDD2")
    for z in [0, 0.5, 1]:
        iwamoto_test.set_metallicity(z)
        assert iwamoto_test.C_12 == 8.99E-03
        assert iwamoto_test.Cl_35 == 7.07E-05
        assert iwamoto_test.Zn_68 == 8.81E-08
        assert iwamoto_test.Zn_64 == 3.10E-05
        with pytest.raises(AttributeError):
            iwamoto_test.U_135
        with pytest.raises(AttributeError):
            iwamoto_test.H_1

def test_make_iwamoto_wdd3():
    iwamoto_test = yields_base.Yields("iwamoto_99_Ia_WDD3")
    for z in [0, 0.5, 1]:
        iwamoto_test.set_metallicity(z)
        assert iwamoto_test.C_12 == 1.66E-02
        assert iwamoto_test.Cl_35 == 5.33E-05
        assert iwamoto_test.Zn_68 == 9.42E-08
        assert iwamoto_test.Zn_64 == 5.76E-05
        with pytest.raises(AttributeError):
            iwamoto_test.U_135
        with pytest.raises(AttributeError):
            iwamoto_test.H_1

def test_make_iwamoto_cdd1():
    iwamoto_test = yields_base.Yields("iwamoto_99_Ia_CDD1")
    for z in [0, 0.5, 1]:
        iwamoto_test.set_metallicity(z)
        assert iwamoto_test.C_12 == 9.93E-03
        assert iwamoto_test.Cl_35 == 9.03E-05
        assert iwamoto_test.Zn_68 == 3.08E-09
        assert iwamoto_test.Zn_64 == 1.87E-06
        with pytest.raises(AttributeError):
            iwamoto_test.U_135
        with pytest.raises(AttributeError):
            iwamoto_test.H_1

def test_make_iwamoto_cdd2():
    iwamoto_test = yields_base.Yields("iwamoto_99_Ia_CDD2")
    for z in [0, 0.5, 1]:
        iwamoto_test.set_metallicity(z)
        assert iwamoto_test.C_12 == 5.08E-03
        assert iwamoto_test.Cl_35 == 6.56E-05
        assert iwamoto_test.Zn_68 == 3.03E-08
        assert iwamoto_test.Zn_64 == 3.96E-05
        with pytest.raises(AttributeError):
            iwamoto_test.U_135
        with pytest.raises(AttributeError):
            iwamoto_test.H_1

def test_met_log():
    """Tests the metallicity log function. Is just like log, but returns a
    fixed value for 0."""
    assert yields_base._metallicity_log(0) == -6
    assert yields_base._metallicity_log(1) == 0
    assert yields_base._metallicity_log(0.01) == -2
    assert yields_base._metallicity_log(100) == 2

    # test with arrays
    assert np.array_equal(yields_base._metallicity_log(np.array([0, 1, 0.01])),
                          np.array([-6, 0, -2]))

def test_normalization_stability(yields_test_case):
    """Once we set the normalization, the total amount of metals should not
    change. Make sure that is the case. """
    yields_test_case.set_metallicity(0)
    total_metals = 10
    yields_test_case.normalize_metals(total_metals)
    assert np.isclose(yields_test_case.ejecta_sum(metal_only=True),
                      total_metals)
    # then change the metallicity
    yields_test_case.set_metallicity(0.2)
    assert np.isclose(yields_test_case.ejecta_sum(metal_only=True),
                      total_metals)
    # then do it again
    yields_test_case.set_metallicity(1)
    assert np.isclose(yields_test_case.ejecta_sum(metal_only=True),
                      total_metals)

def test_nomoto_parser():
    """Test the funciton that takes the name and element from the Nomoto file
    and puts it in the right format that we want."""
    assert yields_base._parse_nomoto_element("01", "p") == "H_1"
    assert yields_base._parse_nomoto_element("02", "d") == "H_2"
    assert yields_base._parse_nomoto_element("09", "Be") == "Be_9"
    assert yields_base._parse_nomoto_element("24", "Na") == "Na_24"
    assert yields_base._parse_nomoto_element("30", "Si") == "Si_30"


# create simple object for testing
@pytest.fixture
def yields_nomoto():
    return yields_base.Yields("nomoto_06_II")

def test_make_nomoto_at_zero_met(yields_nomoto):
    """Test that the Nomoto yields return the correct values. 
    
    I will test at all the given metallicities in the paper, to make sure it 
    works at the values in the table. I will put all the metallicity points
    in the same function, to make sure that is working correctly too."""
    yields_nomoto.set_metallicity(0)
    assert yields_nomoto.H_1 == 3.28E-02
    assert yields_nomoto.H_2 == 5.76E-18
    assert yields_nomoto.Si_28 == 8.11E-04
    assert yields_nomoto.Ca_48 == 8.04E-15
    assert yields_nomoto.Zn_64 == 6.32E-07
    assert yields_nomoto.Ge_74 == 1.33E-14
    with pytest.raises(AttributeError):
        yields_nomoto.U_135

    yields_nomoto.set_metallicity(0.001)
    assert yields_nomoto.H_1 == 3.14E-02
    assert yields_nomoto.H_2 == 2.21E-15
    assert yields_nomoto.Si_28 == 7.09E-04
    assert yields_nomoto.Ca_48 == 6.91E-10
    assert yields_nomoto.Zn_64 == 5.74E-07
    assert yields_nomoto.Ge_74 == 2.18E-08
    with pytest.raises(AttributeError):
        yields_nomoto.U_135

    yields_nomoto.set_metallicity(0.004)
    assert yields_nomoto.H_1 == 2.96E-02
    assert yields_nomoto.H_2 == 1.97E-16
    assert yields_nomoto.Si_28 == 6.17E-04
    assert yields_nomoto.Ca_48 == 2.93E-09
    assert yields_nomoto.Zn_64 == 5.07E-07
    assert yields_nomoto.Ge_74 == 1.35E-07
    with pytest.raises(AttributeError):
        yields_nomoto.U_135

    yields_nomoto.set_metallicity(0.02)
    assert yields_nomoto.H_1 == 2.45E-02
    assert yields_nomoto.H_2 == 5.34E-16
    assert yields_nomoto.Si_28 == 4.55E-04
    assert yields_nomoto.Ca_48 == 1.07E-08
    assert yields_nomoto.Zn_64 == 4.43E-07
    assert yields_nomoto.Ge_74 == 7.93E-07
    with pytest.raises(AttributeError):
        yields_nomoto.U_135

def test_make_nomoto_interpolation_range(yields_nomoto):
    """Tests that the interpolation is returning values in the range we need."""
    # first just test that the values are in the right range (ie between the
    # abundances of the metallicities that span the metallicity used.
    yields_nomoto.set_metallicity(0.002)
    assert 3.14E-2 > yields_nomoto.H_1 > 2.96E-2
    assert 7.09E-4 > yields_nomoto.Si_28 > 6.17E-4
    assert 3.34E-5 > yields_nomoto.Ca_40 > 3.02E-5

    # try a different metallicity value
    yields_nomoto.set_metallicity(0.01)
    assert 2.96E-2 > yields_nomoto.H_1 > 2.45E-2
    assert 6.17E-4 > yields_nomoto.Si_28 > 4.55E-4
    assert 3.02E-5 > yields_nomoto.Ca_40 > 2.39E-5

def test_make_nomoto_interpolation_values(yields_nomoto):
    """Tests that the interpolation is working correctly by directly testing 
       values, not just checking their range."""
    # I want to get a metallicity directly in between in log space, which can
    # be gotten using the logspace function
    middle = np.logspace(yields_base._metallicity_log(0),
                         yields_base._metallicity_log(0.001), 3)[1]  # get middle val
    yields_nomoto.set_metallicity(middle)
    assert np.isclose(yields_nomoto.H_1, np.mean([3.28E-2, 3.14E-2]))
    assert np.isclose(yields_nomoto.Ca_46, np.mean([5.69E-14, 2.06E-10]))
    assert np.isclose(yields_nomoto.Ge_74, np.mean([1.33E-14, 2.18E-8]))

    # then repeat for a different metallicity
    middle = np.logspace(yields_base._metallicity_log(0.004),
                         yields_base._metallicity_log(0.02), 3)[1]  # get middle val
    yields_nomoto.set_metallicity(middle)
    assert np.isclose(yields_nomoto.H_1, np.mean([2.96E-2, 2.45E-2]))
    assert np.isclose(yields_nomoto.Ca_46, np.mean([8.71E-10, 3.60E-9]))
    assert np.isclose(yields_nomoto.Ge_74, np.mean([1.35E-7, 7.93E-7]))

def test_metallicity_outside_range_nomoto(yields_nomoto):
    """Tests what happens when the metallicity is outside the range the 
    models span. I will assert that it should be the same as the yields of the
    model that is at the extreme. """
    yields_nomoto.set_metallicity(0.99)
    assert yields_nomoto.H_1 == 2.45E-2
    assert yields_nomoto.H_2 == 5.34E-16
    assert yields_nomoto.O_16 == 6.14E-3
    assert yields_nomoto.Al_27 == 6.53E-5
    assert yields_nomoto.Fe_58 == 2.15E-6
    assert yields_nomoto.Fe_54 == 1.13E-5

def test_metallicity_sums(yields_nomoto):
    for Z in [0, 0.001, 0.3, 0.5, 0.9]:
        yields_nomoto.set_metallicity(Z)
        calciums = [value for key, value in yields_nomoto.abundances.items()
                    if "Ca_" in key]
        assert np.isclose(np.sum(calciums), yields_nomoto.Ca)

        sodiums = [value for key, value in yields_nomoto.abundances.items()
                   if "Na_" in key]
        assert np.isclose(np.sum(sodiums), yields_nomoto.Na)


def test_metallicity_sums_with_normalization(yields_nomoto):
    yields_nomoto.normalize_metals(100000)
    for Z in [0, 0.001, 0.3, 0.5, 0.9]:
        yields_nomoto.set_metallicity(Z)
        calciums = [value for key, value in yields_nomoto.abundances.items()
                    if "Ca_" in key]
        assert np.isclose(np.sum(calciums), yields_nomoto.Ca)

        sodiums = [value for key, value in yields_nomoto.abundances.items()
                   if "Na_" in key]
        assert np.isclose(np.sum(sodiums), yields_nomoto.Na)

def test_parse_nomoto_individual_element():
    """Test whether this parsing works for everything"""
    assert yields_base._parse_nomoto_individual_element("p") == "H_1"
    assert yields_base._parse_nomoto_individual_element("d") == "H_2"
    assert yields_base._parse_nomoto_individual_element("3He") == "He_3"
    assert yields_base._parse_nomoto_individual_element("18O") == "O_18"
    assert yields_base._parse_nomoto_individual_element("28Si") == "Si_28"

def test_individual_nomoto_mass_13():
    individual = yields_base.Yields("nomoto_06_II_13")

    assert individual.mass_cuts[0] == 1.57
    assert individual.mass_cuts[0.001] == 1.65
    assert individual.mass_cuts[0.004] == 1.61
    assert individual.mass_cuts[0.02] == 1.60

    individual.set_metallicity(0)
    assert individual.O_18 == 5.79E-8
    assert individual.Ga_71 == 8.53E-15

    individual.set_metallicity(0.001)
    assert individual.C_12 == 1.07E-1
    assert individual.Si_30 == 1.85E-3

    individual.set_metallicity(0.004)
    assert individual.O_16 == 3.85E-1
    assert individual.Ca_40 == 3.91E-3

    individual.set_metallicity(0.02)
    assert individual.B_11 == 4.28E-10
    assert individual.Al_26 == 2.13E-5

def test_individual_nomoto_mass_18():
    individual = yields_base.Yields("nomoto_06_II_18")

    assert individual.mass_cuts[0] == 1.65
    assert individual.mass_cuts[0.001] == 1.70
    assert individual.mass_cuts[0.004] == 1.61
    assert individual.mass_cuts[0.02] == 1.58

    individual.set_metallicity(0)
    assert individual.O_18 == 4.63E-6
    assert individual.Ga_71 == 1.84E-14

    individual.set_metallicity(0.001)
    assert individual.C_12 == 1.29E-1
    assert individual.Si_30 == 5.34E-4

    individual.set_metallicity(0.004)
    assert individual.O_16 == 5.21E-1
    assert individual.Ca_40 == 6.12E-3

    individual.set_metallicity(0.02)
    assert individual.B_11 == 6.41E-10
    assert individual.Al_26 == 3.69E-5

def test_individual_nomoto_mass_25():
    individual = yields_base.Yields("nomoto_06_II_25")

    assert individual.mass_cuts[0] == 1.92
    assert individual.mass_cuts[0.001] == 1.91
    assert individual.mass_cuts[0.004] == 1.68
    assert individual.mass_cuts[0.02] == 1.69

    individual.set_metallicity(0)
    assert individual.O_18 == 6.75E-7
    assert individual.Ga_71 == 2.24E-13

    individual.set_metallicity(0.001)
    assert individual.C_12 == 2.15E-1
    assert individual.Si_30 == 2.75E-4

    individual.set_metallicity(0.004)
    assert individual.O_16 == 2.20
    assert individual.Ca_40 == 3.77E-3

    individual.set_metallicity(0.02)
    assert individual.B_11 == 6.77E-10
    assert individual.Al_26 == 8.67E-5

def test_individual_nomoto_mass_40():
    individual = yields_base.Yields("nomoto_06_II_40")

    assert individual.mass_cuts[0] == 2.89
    assert individual.mass_cuts[0.001] == 3.17
    assert individual.mass_cuts[0.004] == 2.81
    assert individual.mass_cuts[0.02] == 2.21

    individual.set_metallicity(0)
    assert individual.O_18 == 2.13E-7
    assert individual.Ga_71 == 1.36E-15

    individual.set_metallicity(0.001)
    assert individual.C_12 == 7.37E-2
    assert individual.Si_30 == 1.01E-2

    individual.set_metallicity(0.004)
    assert individual.O_16 == 7.96
    assert individual.Ca_40 == 2.83E-2

    individual.set_metallicity(0.02)
    assert individual.B_11 == 3.22E-14
    assert individual.Al_26 == 6.64E-5

def test_nomoto_hn_20():
    individual = yields_base.Yields("nomoto_06_II_20_hn")

    assert individual.mass_cuts[0] == 1.87
    assert individual.mass_cuts[0.001] == 2.20
    assert individual.mass_cuts[0.004] == 2.20
    assert individual.mass_cuts[0.02] == 1.74

    individual.set_metallicity(0)
    assert individual.He_3 == 4.76E-5
    assert individual.B_10 == 1.95E-19

    individual.set_metallicity(0.001)
    assert individual.H_1 == 8.43
    assert individual.Ne_20 == 4.56E-1

    individual.set_metallicity(0.004)
    assert individual.Si_29 == 1.78E-3
    assert individual.P_31 == 4.68E-4

    individual.set_metallicity(0.02)
    assert individual.O_16 == 9.80E-1
    assert individual.Al_26 == 1.81E-5


def test_nomoto_hn_25():
    individual = yields_base.Yields("nomoto_06_II_25_hn")

    assert individual.mass_cuts[0] == 2.74
    assert individual.mass_cuts[0.001] == 2.09
    assert individual.mass_cuts[0.004] == 1.96
    assert individual.mass_cuts[0.02] == 2.03

    individual.set_metallicity(0)
    assert individual.He_3 == 2.11E-4
    assert individual.B_10 == 7.45E-14

    individual.set_metallicity(0.001)
    assert individual.H_1 == 9.80
    assert individual.Ne_20 == 1.05

    individual.set_metallicity(0.004)
    assert individual.Si_29 == 2.27E-3
    assert individual.P_31 == 5.29E-4

    individual.set_metallicity(0.02)
    assert individual.O_16 == 2.18
    assert individual.Al_26 == 6.23E-5


def test_nomoto_hn_30():
    individual = yields_base.Yields("nomoto_06_II_30_hn")

    assert individual.mass_cuts[0] == 3.13
    assert individual.mass_cuts[0.001] == 2.54
    assert individual.mass_cuts[0.004] == 3.98
    assert individual.mass_cuts[0.02] == 2.97

    individual.set_metallicity(0)
    assert individual.He_3 == 2.06E-4
    assert individual.B_10 == 1.05E-14

    individual.set_metallicity(0.001)
    assert individual.H_1 == 1.10E1
    assert individual.Ne_20 == 1.04

    individual.set_metallicity(0.004)
    assert individual.Si_29 == 4.86E-3
    assert individual.P_31 == 1.35E-3

    individual.set_metallicity(0.02)
    assert individual.O_16 == 2.74
    assert individual.Al_26 == 4.84E-5


def test_nomoto_hn_40():
    individual = yields_base.Yields("nomoto_06_II_40_hn")

    assert individual.mass_cuts[0] == 5.43
    assert individual.mass_cuts[0.001] == 5.41
    assert individual.mass_cuts[0.004] == 4.41
    assert individual.mass_cuts[0.02] == 2.63

    individual.set_metallicity(0)
    assert individual.He_3 == 2.56E-5
    assert individual.B_10 == 9.41E-15

    individual.set_metallicity(0.001)
    assert individual.H_1 == 1.29E1
    assert individual.Ne_20 == 1.83E-1

    individual.set_metallicity(0.004)
    assert individual.Si_29 == 6.63E-3
    assert individual.P_31 == 2.00E-3

    individual.set_metallicity(0.02)
    assert individual.O_16 == 7.05
    assert individual.Al_26 == 9.70E-5

def test_nomoto_error_checking():
    with pytest.raises(ValueError):
        yields_base.Yields("nomoto_06_II_12_hn")
    with pytest.raises(ValueError):
        yields_base.Yields("nomoto_06_II_19")
    with pytest.raises(ValueError):
        yields_base.Yields("nomoto_06_12")

# ----------------------------------------------------------

# Testing Kobayashi 2006
# Tests are almost identical to Nomoto

# ----------------------------------------------------------
def test_individual_kobayashi_mass_13():
    individual = yields_base.Yields("kobayashi_06_II_13")

    assert individual.mass_cuts[0] == 1.57
    assert individual.mass_cuts[0.001] == 1.65
    assert individual.mass_cuts[0.004] == 1.61
    assert individual.mass_cuts[0.02] == 1.60

    assert individual.wind_ejecta[0] == 0
    assert individual.wind_ejecta[0.001] == pytest.approx(0.07)
    assert individual.wind_ejecta[0.004] == pytest.approx(0.14)
    assert individual.wind_ejecta[0.02] == pytest.approx(0.27)

    for z in [0, 0.001, 0.004, 0.02]:
        assert individual.energy_erg[z] == pytest.approx(1E51)

    individual.set_metallicity(0)
    assert individual.O_18 == 5.79E-8
    assert individual.Ga_71 == 8.53E-15

    individual.set_metallicity(0.001)
    assert individual.C_12 == 1.07E-1
    assert individual.Si_30 == 1.85E-3

    individual.set_metallicity(0.004)
    assert individual.O_16 == 3.85E-1
    assert individual.Ca_40 == 3.91E-3

    individual.set_metallicity(0.02)
    assert individual.B_11 == 4.28E-10
    assert individual.Al_27 == 1.50E-3

def test_individual_kobayashi_mass_18():
    individual = yields_base.Yields("kobayashi_06_II_18")

    assert individual.mass_cuts[0] == 1.65
    assert individual.mass_cuts[0.001] == 1.70
    assert individual.mass_cuts[0.004] == 1.61
    assert individual.mass_cuts[0.02] == 1.58

    assert individual.wind_ejecta[0] == 0
    assert individual.wind_ejecta[0.001] == pytest.approx(0.16)
    assert individual.wind_ejecta[0.004] == pytest.approx(1.41)
    assert individual.wind_ejecta[0.02] == pytest.approx(1.24)

    for z in [0, 0.001, 0.004, 0.02]:
        assert individual.energy_erg[z] == pytest.approx(1E51)

    individual.set_metallicity(0)
    assert individual.O_18 == 4.63E-6
    assert individual.Ga_71 == 1.84E-14

    individual.set_metallicity(0.001)
    assert individual.C_12 == 1.30E-1
    assert individual.Si_30 == 5.34E-4

    individual.set_metallicity(0.004)
    assert individual.O_16 == 5.21E-1
    assert individual.Ca_40 == 6.12E-3

    individual.set_metallicity(0.02)
    assert individual.B_11 == 6.41E-10
    assert individual.Al_27 == 1.00E-2

def test_individual_kobayashi_mass_25():
    individual = yields_base.Yields("kobayashi_06_II_25")

    assert individual.mass_cuts[0] == 1.92
    assert individual.mass_cuts[0.001] == 1.91
    assert individual.mass_cuts[0.004] == 1.68
    assert individual.mass_cuts[0.02] == 1.69

    assert individual.wind_ejecta[0] == 0
    assert individual.wind_ejecta[0.001] == pytest.approx(0.58)
    assert individual.wind_ejecta[0.004] == pytest.approx(0.97)
    assert individual.wind_ejecta[0.02] == pytest.approx(3.37)

    for z in [0, 0.001, 0.004, 0.02]:
        assert individual.energy_erg[z] == pytest.approx(1E51)

    individual.set_metallicity(0)
    assert individual.O_18 == 6.75E-7
    assert individual.Ga_71 == 2.24E-13

    individual.set_metallicity(0.001)
    assert individual.C_12 == 2.15E-1
    assert individual.Si_30 == 2.75E-4

    individual.set_metallicity(0.004)
    assert individual.O_16 == 2.20
    assert individual.Ca_40 == 3.77E-3

    individual.set_metallicity(0.02)
    assert individual.B_11 == 6.77E-10
    assert individual.Al_27 == 2.70E-2

def test_individual_kobayashi_mass_40():
    individual = yields_base.Yields("kobayashi_06_II_40")

    assert individual.mass_cuts[0] == 2.89
    assert individual.mass_cuts[0.001] == 3.17
    assert individual.mass_cuts[0.004] == 2.81
    assert individual.mass_cuts[0.02] == 2.21

    assert individual.wind_ejecta[0] == 0
    assert individual.wind_ejecta[0.001] == pytest.approx(2.19)
    assert individual.wind_ejecta[0.004] == pytest.approx(7.07)
    assert individual.wind_ejecta[0.02] == pytest.approx(18.17)

    for z in [0, 0.001, 0.004, 0.02]:
        assert individual.energy_erg[z] == pytest.approx(1E51)

    individual.set_metallicity(0)
    assert individual.O_18 == 2.13E-7
    assert individual.Ga_71 == 1.36E-15

    individual.set_metallicity(0.001)
    assert individual.C_12 == 7.37E-2
    assert individual.Si_30 == 1.01E-2

    individual.set_metallicity(0.004)
    assert individual.O_16 == 7.96
    assert individual.Ca_40 == 2.83E-2

    individual.set_metallicity(0.02)
    assert individual.B_11 == 3.22E-14
    assert individual.Al_27 == 8.30E-2

def test_kobayashi_hn_20():
    individual = yields_base.Yields("kobayashi_06_II_20_hn")

    assert individual.mass_cuts[0] == 1.88
    assert individual.mass_cuts[0.001] == 2.24
    assert individual.mass_cuts[0.004] == 2.23
    assert individual.mass_cuts[0.02] == 1.77

    assert individual.wind_ejecta[0] == 0
    assert individual.wind_ejecta[0.001] == pytest.approx(0.29)
    assert individual.wind_ejecta[0.004] == pytest.approx(0.49)
    assert individual.wind_ejecta[0.02] == pytest.approx(1.64)

    for z in [0, 0.001, 0.004, 0.02]:
        assert individual.energy_erg[z] == pytest.approx(1E52)

    individual.set_metallicity(0)
    assert individual.He_3 == 4.76E-5
    assert individual.B_10 == 1.95E-19

    individual.set_metallicity(0.001)
    assert individual.H_1 == 8.43
    assert individual.Ne_20 == 4.56E-1

    individual.set_metallicity(0.004)
    assert individual.Si_29 == 1.78E-3
    assert individual.P_31 == 4.68E-4

    individual.set_metallicity(0.02)
    assert individual.O_16 == 9.80E-1
    assert individual.Al_27 == 8.56E-3


def test_kobayashi_hn_25():
    individual = yields_base.Yields("kobayashi_06_II_25_hn")

    assert individual.mass_cuts[0] == 2.80
    assert individual.mass_cuts[0.001] == 2.15
    assert individual.mass_cuts[0.004] == 1.97
    assert individual.mass_cuts[0.02] == 2.09

    assert individual.wind_ejecta[0] == 0
    assert individual.wind_ejecta[0.001] == pytest.approx(0.55)
    assert individual.wind_ejecta[0.004] == pytest.approx(0.98)
    assert individual.wind_ejecta[0.02] == pytest.approx(3.37)

    for z in [0, 0.001, 0.004, 0.02]:
        assert individual.energy_erg[z] == pytest.approx(1E52)

    individual.set_metallicity(0)
    assert individual.He_3 == 2.11E-4
    assert individual.B_10 == 7.45E-14

    individual.set_metallicity(0.001)
    assert individual.H_1 == 9.80
    assert individual.Ne_20 == 1.05

    individual.set_metallicity(0.004)
    assert individual.Si_29 == 2.27E-3
    assert individual.P_31 == 5.29E-4

    individual.set_metallicity(0.02)
    assert individual.O_16 == 2.18
    assert individual.Al_27 == 2.34E-2


def test_kobayashi_hn_30():
    individual = yields_base.Yields("kobayashi_06_II_30_hn")

    assert individual.mass_cuts[0] == 3.27
    assert individual.mass_cuts[0.001] == 2.57
    assert individual.mass_cuts[0.004] == 4.05
    assert individual.mass_cuts[0.02] == 3.05

    assert individual.wind_ejecta[0] == 0
    assert individual.wind_ejecta[0.001] == pytest.approx(0.95)
    assert individual.wind_ejecta[0.004] == pytest.approx(2.45)
    assert individual.wind_ejecta[0.02] == pytest.approx(5.42)

    for z in [0, 0.001, 0.004, 0.02]:
        assert individual.energy_erg[z] == pytest.approx(2E52)

    individual.set_metallicity(0)
    assert individual.He_3 == 2.06E-4
    assert individual.B_10 == 1.05E-14

    individual.set_metallicity(0.001)
    assert individual.H_1 == 1.11E1
    assert individual.Ne_20 == 1.05

    individual.set_metallicity(0.004)
    assert individual.Si_29 == 4.86E-3
    assert individual.P_31 == 1.35E-3

    individual.set_metallicity(0.02)
    assert individual.O_16 == 2.74
    assert individual.Al_27 == 2.37E-2


def test_kobayashi_hn_40():
    individual = yields_base.Yields("kobayashi_06_II_40_hn")

    assert individual.mass_cuts[0] == 5.53
    assert individual.mass_cuts[0.001] == 5.52
    assert individual.mass_cuts[0.004] == 4.51
    assert individual.mass_cuts[0.02] == 2.67

    assert individual.wind_ejecta[0] == 0
    assert individual.wind_ejecta[0.001] == pytest.approx(2.18)
    assert individual.wind_ejecta[0.004] == pytest.approx(7.07)
    assert individual.wind_ejecta[0.02] == pytest.approx(18.16)

    for z in [0, 0.001, 0.004, 0.02]:
        assert individual.energy_erg[z] == pytest.approx(3E52)

    individual.set_metallicity(0)
    assert individual.He_3 == 2.57E-5
    assert individual.B_10 == 9.41E-15

    individual.set_metallicity(0.001)
    assert individual.H_1 == 1.29E1
    assert individual.Ne_20 == 1.83E-1

    individual.set_metallicity(0.004)
    assert individual.Si_29 == 6.63E-3
    assert individual.P_31 == 2.00E-3

    individual.set_metallicity(0.02)
    assert individual.O_16 == 7.05
    assert individual.Al_27 == 7.20E-2

def test_kobayashi_error_checking():
    with pytest.raises(ValueError):
        yields_base.Yields("kobayashi_06_II_12_hn")
    with pytest.raises(ValueError):
        yields_base.Yields("kobayashi_06_II_19")
    with pytest.raises(ValueError):
        yields_base.Yields("kobayashi_06_12")

@pytest.mark.parametrize("tag", ["13", "15", "18", "20", "25", "30", "40",
                                 "20_hn", "25_hn", "30_hn", "40_hn"])
@pytest.mark.parametrize("z", [0, 0.001, 0.004, 0.02])
def test_kobayashi_mass_conservation(tag, z):
    model_name = "kobayashi_06_II_{}".format(tag)
    model = yields_base.Yields(model_name)
    model.set_metallicity(z)

    ejecta_sum = model.ejecta_sum(metal_only=False)
    model_ejecta_sum = model.total_end_ejecta[z]
    # have tolerance of two decimal places, since that's all the mass cut
    # and m_final are reported to. The actual tests don't pass, indicating that
    # the tables are a little untrustworthy
    assert pytest.approx(model_ejecta_sum, abs=0.01) == ejecta_sum

# ----------------------------------------------------------

# Testing WW 95

# ----------------------------------------------------------

def test_ww_individual_11():
    individual = yields_base.Yields("ww_95_II_11A")

    individual.set_metallicity(0.02)
    assert individual.O_16 == 1.36E-1
    assert individual.H_1 == 5.59
    assert individual.Si_28 == 2.17E-2

def test_ww_individual_12():
    individual = yields_base.Yields("ww_95_II_12A")

    individual.set_metallicity(0.02)
    assert individual.Li_7 == 2.18E-7
    assert individual.Mg_26 == 1.69E-3

    individual.set_metallicity(0.002)
    assert individual.O_16 == 1.45E-1
    assert individual.Ar_40 == 1.16E-7

    individual.set_metallicity(0.0002)
    assert individual.N_14 == 3.33E-4
    assert individual.Ti_44 == 6.13E-5

    individual.set_metallicity(10**-4 * 0.02)
    assert individual.Co_55 == 1.99E-4
    assert individual.Ga_67 == 3.72E-9

    individual.set_metallicity(0)
    assert individual.Cl_36 == 7.92E-8
    assert individual.Fe_54 == 3.35E-4

def test_ww_individual_13():
    individual = yields_base.Yields("ww_95_II_13A")

    individual.set_metallicity(0.02)
    assert individual.Li_7 == 1.98E-7
    assert individual.Mg_26 == 3.40E-3

    individual.set_metallicity(0.002)
    assert individual.O_16 == 2.90E-1
    assert individual.Ar_40 == 1.45E-7

    individual.set_metallicity(0.0002)
    assert individual.N_14 == 4.36E-4
    assert individual.Ti_44 == 6.40E-5

    individual.set_metallicity(10 ** -4 * 0.02)
    assert individual.Co_55 == 2.65E-4
    assert individual.Ga_67 == 3.86E-9

    individual.set_metallicity(0)
    assert individual.Cl_36 == 1.25E-7
    assert individual.Fe_54 == 5.40E-4

def test_ww_individual_15():
    individual = yields_base.Yields("ww_95_II_15A")

    individual.set_metallicity(0.02)
    assert individual.Li_7 == 2.07E-7
    assert individual.Mg_26 == 6.50E-3

    individual.set_metallicity(0.002)
    assert individual.O_16 == 5.55E-1
    assert individual.Ar_40 == 2.05E-7

    individual.set_metallicity(0.0002)
    assert individual.N_14 == 5.43E-4
    assert individual.Ti_44 == 9.06E-5

    individual.set_metallicity(10 ** -4 * 0.02)
    assert individual.Co_55 == 1.53E-4
    assert individual.Ga_67 == 5.33E-9

    individual.set_metallicity(0)
    assert individual.Cl_36 == 2.93E-7
    assert individual.Fe_54 == 7.42E-4

def test_ww_individual_18():
    individual = yields_base.Yields("ww_95_II_18A")

    individual.set_metallicity(0.02)
    assert individual.Li_7 == 2.73E-7
    assert individual.Mg_26 == 9.85E-3

    individual.set_metallicity(0.002)
    assert individual.O_16 == 9.94E-1
    assert individual.Ar_40 == 2.96E-7

    individual.set_metallicity(0.0002)
    assert individual.N_14 == 6.24E-4
    assert individual.Ti_44 == 4.95E-5

    individual.set_metallicity(10 ** -4 * 0.02)
    assert individual.Co_55 == 5.23E-4
    assert individual.Ga_67 == 5.58E-9

    individual.set_metallicity(0)
    assert individual.Cl_36 == 1.61E-14
    assert individual.Fe_54 == 1.09E-20

def test_ww_individual_19():
    individual = yields_base.Yields("ww_95_II_19A")

    individual.set_metallicity(0.02)
    assert individual.Li_7 == 2.51E-7
    assert individual.Mg_26 == 1.14E-2

def test_ww_individual_20():
    individual = yields_base.Yields("ww_95_II_20A")

    individual.set_metallicity(0.02)
    assert individual.Li_7 == 2.51E-7
    assert individual.Mg_26 == 1.05E-2

    individual.set_metallicity(0.002)
    assert individual.O_16 == 1.52
    assert individual.Ar_40 == 3.58E-7

    individual.set_metallicity(0.0002)
    assert individual.N_14 == 6.81E-4
    assert individual.Ti_44 == 1.24E-5

    individual.set_metallicity(10 ** -4 * 0.02)
    assert individual.Co_55 == 5.23E-4
    assert individual.Ga_67 == 1.78E-10

    individual.set_metallicity(0)
    assert individual.Cl_36 == 2.14E-14
    assert individual.Fe_54 == 1.13E-23

def test_ww_individual_22():
    individual = yields_base.Yields("ww_95_II_22A")

    individual.set_metallicity(0.02)
    assert individual.Li_7 == 2.17E-7
    assert individual.Mg_26 == 1.28E-2

    individual.set_metallicity(0.002)
    assert individual.O_16 == 2.12
    assert individual.Ar_40 == 3.48E-7

    individual.set_metallicity(0.0002)
    assert individual.N_14 == 7.77E-4
    assert individual.Ti_44 == 1.46E-5

    individual.set_metallicity(10 ** -4 * 0.02)
    assert individual.Co_55 == 4.71E-4
    assert individual.Ga_67 == 1.06E-9

    individual.set_metallicity(0)
    assert individual.Cl_36 == 1.71E-6
    assert individual.Fe_54 == 2.52E-3

def test_ww_individual_25():
    individual = yields_base.Yields("ww_95_II_25A")

    individual.set_metallicity(0.02)
    assert individual.Li_7 == 2.40E-7
    assert individual.Mg_26 == 3.25E-2

    individual.set_metallicity(0.002)
    assert individual.O_16 == 2.90
    assert individual.Ar_40 == 3.92E-7

    individual.set_metallicity(0.0002)
    assert individual.N_14 == 9.29E-4
    assert individual.Ti_44 == 1.11E-4

    individual.set_metallicity(10 ** -4 * 0.02)
    assert individual.Co_55 == 5.20E-4
    assert individual.Ga_67 == 1.71E-8

    individual.set_metallicity(0)
    assert individual.Cl_36 == 8.42E-13
    assert individual.Fe_54 == 1.56E-17

def test_ww_individual_25B():
    individual = yields_base.Yields("ww_95_II_25B")

    individual.set_metallicity(0)
    assert individual.Cl_36 == 9.00E-7
    assert individual.Fe_54 == 2.69E-3

def test_ww_individual_30A():
    individual = yields_base.Yields("ww_95_II_30A")

    individual.set_metallicity(0.02)
    assert individual.O_16 == 3.65
    assert individual.Ca_41 == 3.42E-6

    individual.set_metallicity(0.002)
    assert individual.O_16 == 4.11
    assert individual.Ar_40 == 9.67E-7

    individual.set_metallicity(0.0002)
    # assert individual.Fe_56 == 1.07E-4
    assert individual.Ge_65 == 2.49E-22

    individual.set_metallicity(10**-4 * 0.02)
    assert individual.Ca_43 == 1.45E-10
    assert individual.Co_61 == 3.18E-10

    individual.set_metallicity(0)
    assert individual.C_12 == 1.09E-1
    assert individual.Al_26 == 6.24E-11

def test_ww_individual_30B():
    individual = yields_base.Yields("ww_95_II_30B")

    individual.set_metallicity(0.02)
    assert individual.O_16 == 4.88
    assert individual.Ca_41 == 1.25E-5

    individual.set_metallicity(0.002)
    assert individual.O_16 == 4.42
    assert individual.Ar_40 == 9.52E-7

    individual.set_metallicity(0.0002)
    # assert individual.Fe_56 == 1.17E-4
    assert individual.Ge_65 == 5.89E-22

    individual.set_metallicity(10 ** -4 * 0.02)
    assert individual.Ca_43 == 4.59E-8
    assert individual.Co_61 == 4.61E-10

    individual.set_metallicity(0)
    assert individual.C_12 == 3.48E-1
    assert individual.Al_26 == 5.92E-5

def test_ww_individual_35A():
    individual = yields_base.Yields("ww_95_II_35A")

    individual.set_metallicity(0.02)
    assert individual.O_16 == 3.07
    assert individual.Ca_41 == 3.60E-6

    individual.set_metallicity(0.002)
    assert individual.O_16 == 3.10
    assert individual.Ar_40 == 1.04E-6

    individual.set_metallicity(0.0002)
    # assert individual.Fe_56 == 1.21E-4
    assert individual.Ge_65 == 8.56E-24

    individual.set_metallicity(10 ** -4 * 0.02)
    assert individual.Ca_43 == 2.50E-9
    assert individual.Co_61 == 5.46E-10

    individual.set_metallicity(0)
    assert individual.C_12 == 9.79E-10
    assert individual.Al_26 == 7.38E-15

def test_ww_individual_35B():
    individual = yields_base.Yields("ww_95_II_35B")

    individual.set_metallicity(0.02)
    assert individual.O_16 == 5.82
    assert individual.Ca_41 == 4.20E-6

    individual.set_metallicity(0.002)
    assert individual.O_16 == 5.78
    assert individual.Ar_40 == 1.47E-6

    individual.set_metallicity(0.0002)
    # assert individual.Fe_56 == 1.32E-4
    assert individual.Ge_65 == 3.28E-23

    individual.set_metallicity(10 ** -4 * 0.02)
    assert individual.Ca_43 == 2.80E-9
    assert individual.Co_61 == 1.09E-9

    individual.set_metallicity(0)
    assert individual.C_12 == 3.49E-1
    assert individual.Al_26 == 6.32E-6

def test_ww_individual_35C():
    individual = yields_base.Yields("ww_95_II_35C")

    individual.set_metallicity(0.02)
    assert individual.O_16 == 6.36
    assert individual.Ca_41 == 1.46E-5

    individual.set_metallicity(0.002)
    assert individual.O_16 == 5.98
    assert individual.Ar_40 == 1.45E-6

    individual.set_metallicity(0.0002)
    # assert individual.Fe_56 == 1.35E-4
    assert individual.Ge_65 == 9.04E-18

    individual.set_metallicity(10 ** -4 * 0.02)
    assert individual.Ca_43 == 4.66E-7
    assert individual.Co_61 == 1.29E-9

    individual.set_metallicity(0)
    assert individual.C_12 == 4.04E-1
    assert individual.Al_26 == 9.51E-5

def test_ww_individual_40A():
    individual = yields_base.Yields("ww_95_II_40A")

    individual.set_metallicity(0.02)
    assert individual.O_16 == 2.36
    assert individual.Ca_41 == 3.54E-6

    individual.set_metallicity(0.002)
    assert individual.O_16 == 2.72
    assert individual.Ar_40 == 1.16E-6

    individual.set_metallicity(0.0002)
    # assert individual.Fe_56 == 1.32E-4
    assert individual.Ge_65 == 2.59E-24

    individual.set_metallicity(10 ** -4 * 0.02)
    assert individual.Ca_43 == 6.92E-12
    assert individual.Co_61 == 2.22E-11

    individual.set_metallicity(0)
    assert individual.C_12 == 5.89E-10
    assert individual.Al_26 == 8.12E-15

def test_ww_individual_40B():
    individual = yields_base.Yields("ww_95_II_40B")

    individual.set_metallicity(0.02)
    assert individual.O_16 == 6.03
    assert individual.Ca_41 == 4.28E-6

    individual.set_metallicity(0.002)
    assert individual.O_16 == 6.25
    assert individual.Ar_40 == 2.38E-6

    individual.set_metallicity(0.0002)
    # assert individual.Fe_56 == 1.33E-4
    assert individual.Ge_65 == 1.99E-23

    individual.set_metallicity(10 ** -4 * 0.02)
    assert individual.Ca_43 == 7.37E-11
    assert individual.Co_61 == 8.74E-10

    individual.set_metallicity(0)
    assert individual.C_12 == 2.85E-1
    assert individual.Al_26 == 1.32E-9

def test_ww_individual_40C():
    individual = yields_base.Yields("ww_95_II_40C")

    individual.set_metallicity(0.02)
    assert individual.O_16 == 7.63
    assert individual.Ca_41 == 1.97E-5

    individual.set_metallicity(0.002)
    assert individual.O_16 == 7.15
    assert individual.Ar_40 == 2.36E-6

    individual.set_metallicity(0.0002)
    # assert individual.Fe_56 == 1.56E-4
    assert individual.Ge_65 == 3.74E-23

    individual.set_metallicity(10 ** -4 * 0.02)
    assert individual.Ca_43 == 6.05E-7
    assert individual.Co_61 == 9.71E-10

    individual.set_metallicity(0)
    assert individual.C_12 == 4.82E-1
    assert individual.Al_26 == 1.50E-4

def test_ww_error_checking():
    with pytest.raises(ValueError):
        yields_base.Yields("ww_95_II_12")
    with pytest.raises(ValueError):
        yields_base.Yields("ww_95_II_12B")
    with pytest.raises(ValueError):
        yields_base.Yields("ww_95_II_16A")
    with pytest.raises(ValueError):
        yields_base.Yields("ww_95_II_40D")

# def test_mass_fractions():
#     """This is harder to test, but we can use the WW tables that report
#     the total mass, H, and He mass, so I can calculate this manually to
#     check. this doesn't work right now because of the post-processing
#     we do to the WW95 yields. """
#
#     # some of these need larger error ranges, since calculating the total metals
#     # directly from the table isn't as accurate as the real calculation here,
#     # since I'm simply taking ejecta - H - He.
#     mod = yields_base.Yields("ww_95_II_40C")
#     assert np.isclose(mod.mass_fraction("Li_7", 0.002), 1.413E-8,
#                       atol=0, rtol=1E-2)
#     assert np.isclose(mod.mass_fraction("Fe", 0.002), 0.0013,
#                       atol=0., rtol=2E-2)
#     assert np.isclose(mod.mass_fraction("O_16", 10**-4*0.02), 0.665, atol=0.01)
#     assert np.isclose(mod.mass_fraction("Ne_20", 0), 0.143, atol=0.001)
#     assert np.isclose(mod.mass_fraction("Al", 0), 5.3E-4, atol=0.1E-4)
#
#     # check the vectorization aspect of this
#     met_values = [0, 10**-4*0.02, 0.0002, 0.002]
#     assert np.allclose(mod.mass_fraction("O_16", met_values),
#                        [0.660, 0.665, 0.666, 0.656], atol=0.01)
#     assert np.allclose(mod.mass_fraction("Ge_65", met_values),
#                        [1.8E-21, 1.19E-26, 3.53E-24, 4.98E-23], atol=0,
#                        rtol=2E-2)
#
#     # check a different model
#     mod = yields_base.Yields("ww_95_II_11A")
#     assert np.isclose(mod.mass_fraction("Li_7", 0.02), 5.19E-7, atol=0,
#                       rtol=1E-1)
#     assert np.isclose(mod.mass_fraction("N", 0.02), 0.087, atol=0.001)

# ----------------------------------------------------------

# Testing NuGrid AGB

# ----------------------------------------------------------
def test_nugrid_1():
    agb = yields_base.Yields("nugrid_1")

    assert agb.total_end_ejecta[0.02]   == pytest.approx(1.0 - 0.5601, abs=1E-4)
    assert agb.total_end_ejecta[0.01]   == pytest.approx(1.0 - 0.5593, abs=1E-4)
    assert agb.total_end_ejecta[0.006]  == pytest.approx(1.0 - 0.5678, abs=1E-4)
    assert agb.total_end_ejecta[0.001]  == pytest.approx(1.0 - 0.5845, abs=1E-4)
    assert agb.total_end_ejecta[0.0001] == pytest.approx(1.0 - 0.5296, abs=1E-4)

    agb.set_metallicity(0.02)
    assert agb.O_16 == 4.232E-03
    assert agb.Ca_42 == 2.227E-07

    agb.set_metallicity(0.01)
    assert agb.O_16 == 2.122E-03
    assert agb.Ar_40 == 5.840E-09

    agb.set_metallicity(0.006)
    assert agb.Fe_56 == 3.430E-05
    assert agb.C_12 == 2.658E-04

    agb.set_metallicity(0.001)
    assert agb.Ca_43 == 4.702E-10
    assert agb.N_14 == 1.983E-05

    agb.set_metallicity(0.0001)
    assert agb.C_12 == 8.253E-03
    assert agb.Mg_26 == 1.321E-07

def test_nugrid_1_65():
    agb = yields_base.Yields("nugrid_1.65")

    assert agb.total_end_ejecta[0.02]   == pytest.approx(1.65 - 0.6501, abs=1E-3)
    assert agb.total_end_ejecta[0.01]   == pytest.approx(1.65 - 0.6207, abs=1E-4)
    assert agb.total_end_ejecta[0.006]  == pytest.approx(1.65 - 0.6140, abs=1E-4)
    assert agb.total_end_ejecta[0.001]  == pytest.approx(1.65 - 0.6169, abs=1E-4)
    assert agb.total_end_ejecta[0.0001] == pytest.approx(1.65 - 0.6182, abs=1E-3)

    agb.set_metallicity(0.02)
    assert agb.O_16 == 9.983E-03
    assert agb.Ca_42 == 5.111E-07

    agb.set_metallicity(0.01)
    assert agb.O_16 == 5.892E-03
    assert agb.Ar_40 == 1.404E-08

    agb.set_metallicity(0.006)
    assert agb.Fe_56 == 8.159E-05
    assert agb.C_12 == 7.696E-03

    agb.set_metallicity(0.001)
    assert agb.Ca_43 == 1.503E-09
    assert agb.N_14 == 1.036E-04

    agb.set_metallicity(0.0001)
    assert agb.C_12 == 2.147E-02
    assert agb.Mg_26 == 3.536E-05

def test_nugrid_2():
    agb = yields_base.Yields("nugrid_2")

    assert agb.total_end_ejecta[0.02]   == pytest.approx(2.0 - 0.6201, abs=1E-4)
    assert agb.total_end_ejecta[0.01]   == pytest.approx(2.0 - 0.6118, abs=1E-4)
    assert agb.total_end_ejecta[0.006]  == pytest.approx(2.0 - 0.6176, abs=1E-4)
    assert agb.total_end_ejecta[0.001]  == pytest.approx(2.0 - 0.6352, abs=1E-4)
    assert agb.total_end_ejecta[0.0001] == pytest.approx(2.0 - 0.6648, abs=1E-4)

    agb.set_metallicity(0.02)
    assert agb.O_16 == 1.904E-02
    assert agb.Ca_42 == 7.698E-07

    agb.set_metallicity(0.01)
    assert agb.O_16 == 1.385E-02
    assert agb.Ar_40 == 2.554E-08

    agb.set_metallicity(0.006)
    assert agb.Fe_56 == 1.081E-04
    assert agb.C_12 == 1.783E-02

    agb.set_metallicity(0.001)
    assert agb.Ca_43 == 2.349E-09
    assert agb.N_14 == 1.710E-04

    agb.set_metallicity(0.0001)
    assert agb.C_12 == 2.356E-02
    assert agb.Mg_26 == 6.155E-05

def test_nugrid_3():
    agb = yields_base.Yields("nugrid_3")

    assert agb.total_end_ejecta[0.02]   == pytest.approx(3.0 - 0.6420, abs=1E-4)
    assert agb.total_end_ejecta[0.01]   == pytest.approx(3.0 - 0.6593, abs=1E-3)
    assert agb.total_end_ejecta[0.006]  == pytest.approx(3.0 - 0.7097, abs=1E-3)
    assert agb.total_end_ejecta[0.001]  == pytest.approx(3.0 - 0.8242, abs=1E-3)
    assert agb.total_end_ejecta[0.0001] == pytest.approx(3.0 - 0.8521, abs=1E-3)

    agb.set_metallicity(0.02)
    assert agb.O_16 == 3.594E-02
    assert agb.Ca_42 == 1.389E-06

    agb.set_metallicity(0.01)
    assert agb.O_16 == 2.228E-02
    assert agb.Ar_40 == 6.248E-08

    agb.set_metallicity(0.006)
    assert agb.Fe_56 == 1.796E-04
    assert agb.C_12 == 1.852E-02

    agb.set_metallicity(0.001)
    assert agb.Ca_43 == 2.629E-09
    assert agb.N_14 == 2.212E-04

    agb.set_metallicity(0.0001)
    assert agb.C_12 == 8.857E-03
    assert agb.Mg_26 == 2.716E-05

def test_nugrid_4():
    agb = yields_base.Yields("nugrid_4")

    assert agb.total_end_ejecta[0.02]   == pytest.approx(4.0 - 0.8180, abs=1E-4)
    assert agb.total_end_ejecta[0.01]   == pytest.approx(4.0 - 0.8415, abs=1E-3)
    assert agb.total_end_ejecta[0.006]  == pytest.approx(4.0 - 0.8611, abs=1E-3)
    assert agb.total_end_ejecta[0.001]  == pytest.approx(4.0 - 0.8936, abs=1E-3)
    assert agb.total_end_ejecta[0.0001] == pytest.approx(4.0 - 0.9045, abs=1E-3)

    agb.set_metallicity(0.02)
    assert agb.O_16 == 3.833E-02
    assert agb.Ca_42 == 1.794E-06

    agb.set_metallicity(0.01)
    assert agb.O_16 == 1.977E-02
    assert agb.Ar_40 == 1.142E-07

    agb.set_metallicity(0.006)
    assert agb.Fe_56 == 2.479E-04
    assert agb.C_12 == 9.032E-03

    agb.set_metallicity(0.001)
    assert agb.Ca_43 == 3.753E-09
    assert agb.N_14 == 7.812E-03

    agb.set_metallicity(0.0001)
    assert agb.C_12 == 1.755E-03
    assert agb.Mg_26 == 7.790E-05

def test_nugrid_5():
    agb = yields_base.Yields("nugrid_5")

    assert agb.total_end_ejecta[0.02]   == pytest.approx(5.0 - 0.8768, abs=1E-3)
    assert agb.total_end_ejecta[0.01]   == pytest.approx(5.0 - 0.9101, abs=1E-3)
    assert agb.total_end_ejecta[0.006]  == pytest.approx(5.0 - 0.9533, abs=1E-4)
    assert agb.total_end_ejecta[0.001]  == pytest.approx(5.0 - 0.9855, abs=1E-3)
    assert agb.total_end_ejecta[0.0001] == pytest.approx(5.0 - 0.9917, abs=1E-3)

    agb.set_metallicity(0.02)
    assert agb.O_16 == 3.911E-02
    assert agb.Ca_42 == 2.195E-06

    agb.set_metallicity(0.01)
    assert agb.O_16 == 1.994E-02
    assert agb.Ar_40 == 1.048E-07

    agb.set_metallicity(0.006)
    assert agb.Fe_56 == 3.209E-04
    assert agb.C_12 == 7.953E-04

    agb.set_metallicity(0.001)
    assert agb.Ca_43 == 4.533E-09
    assert agb.N_14 == 6.323E-03

    agb.set_metallicity(0.0001)
    assert agb.C_12 == 6.948E-04
    assert agb.Mg_26 == 1.413E-05

def test_nugrid_6():
    agb = yields_base.Yields("nugrid_6")

    assert agb.total_end_ejecta[0.02]   == pytest.approx(6.0 - 0.9653, abs=2E-4)
    assert agb.total_end_ejecta[0.01]   == pytest.approx(6.0 - 0.9938, abs=1E-3)
    assert agb.total_end_ejecta[0.006]  == pytest.approx(6.0 - 1.053,  abs=1E-3)
    assert agb.total_end_ejecta[0.001]  == pytest.approx(6.0 - 1.122,  abs=1E-3)
    assert agb.total_end_ejecta[0.0001] == pytest.approx(6.0 - 1.125,  abs=1E-3)

    agb.set_metallicity(0.02)
    assert agb.O_16 == 3.782E-02
    assert agb.Ca_42 == 2.592E-06

    agb.set_metallicity(0.01)
    assert agb.O_16 == 1.935E-02
    assert agb.Ar_40 == 1.408E-07

    agb.set_metallicity(0.006)
    assert agb.Fe_56 == 3.918E-04
    assert agb.C_12 == 1.313E-03

    agb.set_metallicity(0.001)
    assert agb.Ca_43 == 5.336E-09
    assert agb.N_14 == 5.149E-03

    agb.set_metallicity(0.0001)
    assert agb.C_12 == 2.315E-04
    assert agb.Mg_26 == 3.306E-06

def test_nugrid_7():
    agb = yields_base.Yields("nugrid_7")

    assert agb.total_end_ejecta[0.02]   == pytest.approx(7.0 - 1.066, abs=1E-3)
    assert agb.total_end_ejecta[0.01]   == pytest.approx(7.0 - 1.114, abs=1E-3)
    assert agb.total_end_ejecta[0.006]  == pytest.approx(7.0 - 1.164, abs=1E-3)
    assert agb.total_end_ejecta[0.001]  == pytest.approx(7.0 - 1.259, abs=1E-3)
    assert agb.total_end_ejecta[0.0001] == pytest.approx(7.0 - 1.272, abs=1E-3)

    agb.set_metallicity(0.02)
    assert agb.O_16 == 4.183E-02
    assert agb.Ca_42 == 3.044E-06

    agb.set_metallicity(0.01)
    assert agb.O_16 == 1.817E-02
    assert agb.Ar_40 == 1.124E-07

    agb.set_metallicity(0.006)
    assert agb.Fe_56 == 4.627E-04
    assert agb.C_12 == 9.541E-04

    agb.set_metallicity(0.001)
    assert agb.Ca_43 == 6.258E-09
    assert agb.N_14 == 7.913E-03


    agb.set_metallicity(0.0001)
    assert agb.C_12 == 1.421E-03
    assert agb.Mg_26 == 3.802E-06

@pytest.mark.parametrize("m", ["1", "1.65", "2", "3", "4", "5", "6", "7"])
def test_nugrid_mass_conservation(m):
    agb = yields_base.Yields("nugrid_{}".format(m))
    for z in agb.metallicity_points:
        agb.set_metallicity(z)
        total_1 = agb.total_end_ejecta[z]
        total_2 = agb.ejecta_sum(metal_only=False)
        assert total_1 == pytest.approx(total_2, rel=1E-3)

# ----------------------------------------------------------

# testing Nomoto 18 Ia

# ----------------------------------------------------------
def test_nomoto_Ia_w7():
    individual = yields_base.Yields("nomoto_18_Ia_W7")

    individual.set_metallicity(0.02)
    assert individual.C_12 == 4.75E-2
    assert individual.C_13 == 5.17E-8
    assert individual.N_14 == 1.1E-5
    assert individual.N_15 == 5.46E-8
    assert individual.O_16 == 5.0E-2
    assert individual.O_17 == 4.6E-6
    assert individual.O_18 == 1.43E-7
    assert individual.Mg_24 == 4.61E-3
    assert individual.Mg_25 == 1.19E-4
    assert individual.Mg_26 == 7.64E-5
    assert individual.S_32 == 7.90E-2
    assert individual.S_33 == 4.70E-4
    assert individual.S_34 == 2.38E-3
    assert individual.S_36 == 2.86E-7
    assert individual.Ca_40 == 9.67E-3
    assert individual.Ca_42 == 2.88E-5
    assert individual.Ca_43 == 1.46E-7
    assert individual.Ca_44 == 9.90E-7
    assert individual.Ca_46 == 3.50E-10
    assert individual.Ca_48 == 2.20E-12
    assert individual.Fe_54 == 0.131
    assert individual.Fe_56 == 0.741
    assert individual.Fe_57 == 2.7E-2
    assert individual.Fe_58 == 6.24E-4
    assert individual.Fe_60 == 1.21E-8
    assert individual.Cl_35 == 1.35E-4

    individual.set_metallicity(0.002)
    assert individual.C_12 == 6.67E-2
    assert individual.C_13 == 1.28E-12
    assert individual.N_14 == 7.83E-10
    assert individual.N_15 == 1.32E-8
    assert individual.O_16 == 9.95E-2
    assert individual.O_17 == 1.32E-11
    assert individual.O_18 == 7.60E-13
    assert individual.Mg_24 == 8.68E-3
    assert individual.Mg_25 == 2.49E-6
    assert individual.Mg_26 == 1.26E-6
    assert individual.S_32 == 7.65E-2
    assert individual.S_33 == 1.18E-4
    assert individual.S_34 == 1.14E-4
    assert individual.S_36 == 6.88E-10
    assert individual.Ca_40 == 1.42E-2
    assert individual.Ca_42 == 1.64E-6
    assert individual.Ca_43 == 5.76E-9
    assert individual.Ca_44 == 1.5E-6
    assert individual.Ca_46 == 4.15E-11
    assert individual.Ca_48 == 1.99E-12
    assert individual.Fe_54 == 0.18
    assert individual.Fe_56 == 0.683
    assert individual.Fe_57 == 1.85E-2
    assert individual.Fe_58 == 5.64E-4
    assert individual.Fe_60 == 1.10E-8
    assert individual.Cl_35 == 1.82E-5

# TODO: handle the cases better for the ww95 models that only have one
#       metallicity